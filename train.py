import argparse
import re

import joblib
import numpy as np
import optuna
import os
import pandas as pd
import random
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
from torch import nn
from torch.optim import Adam
import torch
from Utils.base_train import batch_sampled_data
from Utils.config import TransformerConfig
from classifier import Classifier
from data_loader import ExperimentConfig
from forecaster import Forecaster
from modules.opt_model import NoamOpt
from sklearn.metrics import classification_report
torch.autograd.set_detect_anomaly(True)

'''
This class is used for training and evaluating the neural classification models
'''


class Train:
    def __init__(self, data: pd.DataFrame,
                 args: argparse,
                 pred_len: int,
                 add_residual: bool,
                 class_weights: list,
                 use_weight: bool,
                 seed: int):
        '''
        Initialize the training class with the specified parameters.

        :param data: The data used for training and evaluation from {oil, sev_weather, us_accident}
        :param args: The arguments entered by the user
        :param pred_len: The length of multi-horizon predictions
        :param add_residual: Whether to augment residuals
        :param class_weights: The weights associated with each class (normal / abnormal)
        :param use_weight: Whether to use weights
        :param seed: The random seed initialization
        '''

        # Configuration specific to each experiment
        config = ExperimentConfig(pred_len, args.exp_name)

        # Data attributes
        self.data = data
        self.len_data = len(data)

        # Access experiment-related information
        self.formatter = config.make_data_formatter()
        self.params = self.formatter.get_experiment_params()
        self.total_time_steps = self.params['total_time_steps']  # Total time steps = len(input) + len(output)
        self.num_encoder_steps = self.params['num_encoder_steps']  # Total number of steps assigned to encoder
        self.column_definition = self.params["column_definition"]  # Definition of each column in the dataset

        # Model settings
        self.add_residual = add_residual  # Whether to add residuals
        self.use_weight = use_weight      # Whether to use weights
        self.class_weights = class_weights  # Weights associated with each class
        self.pred_len = pred_len   # Length of future predictions
        self.seed = seed           # Random seed for reproducibility purposes
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")   # Use GPU if available
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)        # Path where the model is saved
        self.model_params = self.formatter.get_default_model_params()           # Parameters of each model
        self.batch_size = self.model_params['minibatch_size'][0]                # Size of batch
        self.num_epochs = self.params['num_epochs']    # Total number of epochs

        # Name of the model
        self.name = "{}{}{}{}_{}".format(args.name, args.exp_name, "_weight" if use_weight else "",
                                       "_add_residual" if self.add_residual else "", self.seed)

        # Save the history of parameters for optuna to avoid repeating on already processed hyper-parameters
        self.param_history = []
        self.eval_results = dict()  # Saving evaluation results
        self.exp_name = args.exp_name  # Name of the experiment
        self.best_model = None  # Save the classification model with the best accuracy score
        self.best_forecasting_model = None  # Save the forecasting model with the best L1 loss
        self.best_res_model = None  # Save the residual model with the best L1 loss
        self.train, self.valid, self.test = self.split_data()  # Split and organize data
        self.run_optuna(args)  # Run optuna (train and validate)
        self.evaluate()        # Evaluate the model

    def define_model(self, vocab_size, src_input_size, d_model, n_heads,
                     stack_size):

        """
        Define the model architecture based on specified parameters.

        :param vocab_size: The size of the vocabulary.
        :param src_input_size: total number of features of the input data.
        :param d_model: The dimensionality of the model's hidden representation.
        :param n_heads: The number of attention heads in the multi-head attention mechanism.
        :param stack_size: The number of layers or stack size for the model.
        :return: The main model, forecasting model, and residual model.
        """

        d_k = int(d_model / n_heads)  # Calculate d_k for Transformer

        # create a TransformerConfig
        config = TransformerConfig(vocab_size=vocab_size,
                                   src_input_size=src_input_size,
                                   d_model=d_model,
                                   d_k=d_k,
                                   n_layers=stack_size,
                                   n_heads=n_heads,
                                   stack_size=stack_size)

        # If add_residual is True, create a forecasting and residual model
        if self.add_residual:
            forecasting_model = Forecaster(config=config, pred_len=self.pred_len,
                                           device=self.device,
                                           seed=self.seed,
                                           residual=False).to(self.device)
            residual_model = Forecaster(config=config, pred_len=self.pred_len,
                                        device=self.device,
                                        seed=self.seed,
                                        residual=True).to(self.device)

        # If add_residual is False, set forecasting_model and residual_model to None
        else:
            forecasting_model = None
            residual_model = None

        # Get the type of data for match for class weights
        _, _, train_y, _ = next(iter(self.train))

        # Convert class_weights to a tensor on the specified device
        if self.use_weight:
            class_weights = torch.tensor(self.class_weights, device=self.device, dtype=train_y.dtype)
        else:
            class_weights = None

        # Create a Classifier model
        model = Classifier(config=config, pred_len=self.pred_len,
                           device=self.device,
                           seed=self.seed, n_classes=2, divide=False,
                           class_weights=class_weights).to(self.device)

        # Return the main model, forecasting_model, and residual_model
        return model, forecasting_model, residual_model

    def split_data(self):
        """
        Split the input data into training, validation, and test sets, and organize them into batches.

        :return: Training, validation, and test datasets, along with the number of training batches.
        """

        # Transform raw data using the formatter
        data = self.formatter.transform_data(self.data)

        # Get the maximum number of samples for calibration from the formatter
        train_max, valid_max = self.formatter.get_num_samples_for_calibration()

        # Store the maximum number of samples for each set
        max_samples = (train_max, valid_max)

        # Batch the sampled data into training, validation, and test sets
        train, valid, test = batch_sampled_data(data, 0.8, max_samples, self.params['total_time_steps'],
                                                self.params['num_encoder_steps'], self.pred_len,
                                                self.params["column_definition"],
                                                self.batch_size)

        # Return the organized datasets and the number of training batches
        return train, valid, test

    def run_optuna(self, args):
        """
        Run the hyperparameter optimization using Optuna.

        :param args: Command-line arguments provided by the user.
        """

        # Create an Optuna study for hyperparameter optimization
        study = optuna.create_study(study_name=args.name,
                                    direction="maximize", pruner=optuna.pruners.HyperbandPruner())
        # parallelize optuna with joblib

        study.optimize(self.objective, n_trials=args.n_trials, n_jobs=1)

        # Get trials that were pruned and completed
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        # Print study statistics
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        # Print details of the best trial
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial):
        """
        Define the objective function for hyperparameter optimization using Optuna.

        :param trial: Optuna trial object representing a single optimization trial.
        :return: The value of the objective function to be maximized (accuracy or other metric).
        """

        # Obtain a batch of training data to define the total number of features
        train_enc, train_dec, train_y, y_true = next(iter(self.train))
        src_input_size = train_enc.shape[2]

        # Get the maximum number of unique classes (vocabulary size)
        vocab_size = max(self.formatter.get_num_classes())

        # Check if the model path exists, if not, create it
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Suggest hyperparameters for the current trial
        d_model = trial.suggest_categorical("d_model", [64])
        stack_size = trial.suggest_categorical("stack_size", [2])
        w_steps = trial.suggest_categorical("w_steps", [4000])
        n_heads = trial.suggest_categorical("n_heads", [8])

        # Check if the current set of hyperparameters has already been tested
        if [d_model, w_steps, stack_size] in self.param_history:
            raise optuna.exceptions.TrialPruned()
        self.param_history.append([d_model, w_steps, stack_size])

        # Define models based on suggested hyperparameters
        model, forecasting_model, residual_model = self.define_model(vocab_size=vocab_size,
                                                                     src_input_size=src_input_size,
                                                                     d_model=d_model,
                                                                     n_heads=n_heads,
                                                                     stack_size=stack_size)

        # Initialize the optimizer for main model
        optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

        # Train forecasting models
        def train_forecaster(predictor, optim, residual=False):

            val_loss_l1 = 1e10

            for e in range(self.num_epochs):

                predictor.train()

                tot_loss = 0

                for trn_enc, trn_dec, trn_y, trn_y_forecasting in self.train:
                    train_y_true = trn_y_forecasting.to(self.device)
                    outputs_forecaster = predictor(trn_enc.to(self.device), trn_dec.to(self.device))

                    if residual:
                        train_y_true = train_y_true - outputs_forecaster

                    l1_loss = nn.L1Loss()(train_y_true, outputs_forecaster)
                    tot_loss += l1_loss.item()

                    optim.zero_grad()
                    l1_loss.backward()
                    optim.step_and_update_lr()

                if e % 5 == 0:
                    print("Train epoch: {}, loss: {:.4f}".format(e, tot_loss))

                predictor.eval()
                test_loss = 0

                for vad_enc, vad_dec, vad_y, vad_y_forecasting in self.valid:
                    valid_y_true = vad_y_forecasting.to(self.device)
                    outputs_forecaster = predictor(vad_enc.to(self.device), vad_dec.to(self.device))
                    if residual:
                        valid_y_true = valid_y_true - outputs_forecaster

                    l1_loss = nn.L1Loss()(valid_y_true, outputs_forecaster)
                    test_loss += l1_loss.item()

                if e % 5 == 0:
                    print("Train epoch: {}, val loss: {:.4f}".format(e, test_loss))

                if test_loss < val_loss_l1:
                    val_loss_l1 = test_loss
                    if not residual:
                        self.best_forecasting_model = predictor
                    else:
                        self.best_res_model = predictor
                    torch.save({'model_state_dict': predictor.state_dict()},
                               os.path.join(self.model_path, "{}_{}".format(self.name, self.seed)))

        # Train forecasting models if residuals are added
        if self.add_residual:

            forecasting_optimizer = NoamOpt(Adam(forecasting_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                                            2, d_model, w_steps)
            train_forecaster(forecasting_model, forecasting_optimizer)
            forecasting_optimizer = NoamOpt(Adam(residual_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                                            2, d_model, w_steps)
            train_forecaster(residual_model, forecasting_optimizer, residual=True)

        val_score = -np.inf

        # Train the main classification model
        for epoch in range(self.num_epochs):

            model.train()

            total_loss = 0
            for train_enc, train_dec, train_y, _ in self.train:

                outputs, loss = model(train_enc.to(self.device), train_dec.to(self.device),
                                      forecasting_model=self.best_forecasting_model,
                                      residual_model=self.best_res_model,
                                      y_true=train_y)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step_and_update_lr()

            if epoch % 5 == 0:
                print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

            model.eval()
            sum_accuracy = 0
            sum_f1 = 0
            i = 0
            for valid_enc, valid_dec, valid_y, _ in self.valid:

                outputs, _ = model(valid_enc.to(self.device), valid_dec.to(self.device),
                                   self.best_forecasting_model,
                                   self.best_res_model)

                valid_y = valid_y.to(self.device, dtype=torch.long).squeeze(1)

                outputs = torch.argmax(outputs, dim=-1)

                valid_y = torch.flatten(valid_y, start_dim=0).cpu().detach().numpy()
                outputs = outputs.reshape(-1).cpu().detach().numpy()

                sum_accuracy += accuracy_score(outputs, valid_y)
                class_weights = {0: self.class_weights[0], 1: self.class_weights[1]}

                if self.use_weight:
                    sum_f1 += f1_score(valid_y, outputs,
                                       average='weighted', labels=np.unique(valid_y),
                                       sample_weight=[class_weights[y] for y in valid_y])
                else:
                    sum_f1 += f1_score(valid_y, outputs)

                i += 1

            accuracy_all = sum_accuracy / i
            f1_all = sum_f1 / i

            if epoch % 5 == 0:
                print("Accuracy: {:.4f}, F1: {:.4f}".format(accuracy_all, f1_all))

            if accuracy_all > val_score:
                val_score = accuracy_all
                self.best_model = model
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join(self.model_path, "{}_{}".format(self.name, self.seed)))

        # Return the accuracy or other metric to be maximized by Optuna
        return val_score

    def evaluate(self):
        """
        Evaluate the trained model on the test dataset and store evaluation metrics.

        :return: None
        """

        # Obtain test labels and total number of batches
        _, _, test_y, _ = next(iter(self.test))
        total_b = len(list(iter(self.test)))

        # Set the best model in evaluation mode
        self.best_model.eval()

        # Initialize arrays to store predictions, prediction probabilities, and true labels
        predictions = np.zeros((total_b, test_y.shape[0], self.pred_len))
        predictions_prob = np.zeros((total_b, test_y.shape[0], self.pred_len))
        test_y_tot = np.zeros((total_b, test_y.shape[0], self.pred_len))

        j = 0

        # Loop through test data batches
        for test_enc, test_dec, test_y, _ in self.test:
            # Get model predictions
            output, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device),
                                        self.best_forecasting_model, self.best_res_model)
            # Store predictions, prediction probabilities, and true labels
            predictions[j, :len(output), :] = torch.argmax(output, dim=-1).squeeze(-1).cpu().detach().numpy()
            predictions_prob[j, :len(output), :] = output[:, :, 1:].squeeze(-1).cpu().detach().numpy()
            test_y_tot[j, :len(output), :] = test_y.cpu().squeeze(-1).detach().numpy()

            j += 1

        # Flatten the arrays for evaluation metrics calculation
        predictions = predictions.reshape(-1)
        test_y_tot = test_y_tot.reshape(-1)

        # Define class weights
        class_weights = {0: self.class_weights[0], 1: self.class_weights[1]}

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_y_tot, predictions)
        precision = precision_score(test_y_tot, predictions, average='weighted', labels=np.unique(test_y_tot),
                                        sample_weight=[class_weights[y] for y in test_y_tot])
        recall = recall_score(test_y_tot, predictions, average='weighted', labels=np.unique(test_y_tot),
                                  sample_weight=[class_weights[y] for y in test_y_tot])

        f1 = (2 * precision * recall) / (precision + recall)

        class_report = classification_report(test_y_tot, predictions,
                                             target_names=["normal", "abnormal"],
                                             digits=3)

        lines = class_report.strip().split('\n')
        lines = lines[1:]
        lines = [re.sub(r'\s+', ',', line) for line in lines]
        lines = [line for line in lines if len(line) > 1]

        # Extract class names and metrics from the report text
        class_names = [line.split(',')[1] for line in lines[0:2]]
        metrics = [line.split(',')[2:] for line in lines]

        # Create a dictionary to store the report data
        report_data = {}
        for i, class_name in enumerate(class_names):
            report_data[class_name] = {
                'precision': float(metrics[i][0]),
                'recall': float(metrics[i][1]),
                'f1-score': float(metrics[i][2]),
                'support': int(metrics[i][3])
            }

        scores_divided = {
            "name": self.name,
            "accuracy": "{:.3f}".format(accuracy),
            "f1_loss": "{:.3f}".format(f1),
            "precision": "{:.3f}".format(precision),
            "recall": "{:.3f}".format(recall),
        }

        # Print and store accuracy
        print("Accuracy {:.4f}".format(accuracy))
        self.eval_results["{}".format(self.name)] = scores_divided

        # Store evaluation results in a csv file

        score_path = "Bosch_Final_scores.csv"
        df = pd.DataFrame.from_dict(self.eval_results, orient='index')

        if os.path.exists(score_path):

            df_old = pd.read_csv(score_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(score_path)

        else:
            df.to_csv(score_path)

def main():
    """
    Main function to set up and execute the training process.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--name", type=str, default="autoformer")
    parser.add_argument("--exp_name", type=str, default='sev_weather')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--n_trials", type=int, default=10)

    args = parser.parse_args()

    # Construct the path to the dataset CSV file
    data_csv_path = "datasets/{}.csv".format(args.exp_name)

    # Read the raw data from the CSV file
    raw_data = pd.read_csv(data_csv_path)

    # Compute class weights for balancing
    train_y_weights = raw_data["target"]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y_weights),
                                                      y=train_y_weights)

    random.seed(1234)
    seeds = np.random.randint(1000, 9999, 3)

    # Loop over different prediction lengths
    for pred_len in [60]:
        for seed in seeds:

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            #Train without weight adjustment and residual augmentation
            Train(raw_data, args, pred_len, add_residual=False, use_weight=False,
                  class_weights=class_weights, seed=seed)

            # Train without residual augmentation
            Train(raw_data, args, pred_len, add_residual=False, use_weight=True,
                  class_weights=class_weights, seed=seed)

            #Train with residual augmentation
            # Train(raw_data, args, pred_len, add_residual=True, use_weight=True,
            #       class_weights=class_weights, seed=seed)


if __name__ == '__main__':
    main()

