import random
import numpy as np
import torch
import torch.nn as nn
from modules.transformer import Transformer


class Classifier(nn.Module):
    def __init__(self, *, divide, n_classes, seed, device, pred_len, config, class_weights):
        """
        Classifier module that uses the Transformer architecture for classification.

        :param divide: Whether to divide the output sequence.
        :param n_classes: Number of classes for classification.
        :param seed: Random seed for reproducibility.
        :param device: Device on which the classifier is executed.
        :param pred_len: Length of the prediction.
        :param config: Configuration parameters for the Transformer model.
        :param class_weights: Class weights for handling class imbalance.
        """
        super(Classifier, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.pred_len = pred_len
        self.device = device
        self.divide = divide

        self.classifier = Transformer(vocab_size=config.vocab_size,
                                      src_input_size=config.src_input_size,
                                      pred_len=pred_len,
                                      device=device,
                                      seed=seed,
                                      d_model=config.d_model,
                                      n_layers=config.n_layers,
                                      n_heads=config.n_heads)

        self.final_projection = nn.Linear(config.d_model * config.src_input_size, n_classes)
        self.norm = nn.LayerNorm(config.d_model * config.src_input_size, device=device)
        self.class_weights = class_weights

    def forward(self, enc_inputs, dec_inputs, forecasting_model=None, residual_model=None,
                y_true=None):
        """
        Forward pass of the classifier.

        :param enc_inputs: Encoder inputs.
        :param dec_inputs: Decoder inputs.
        :param forecasting_model: Forecasting model for auxiliary prediction.
        :param residual_model: Residual model for auxiliary prediction.
        :param y_true: True labels for calculating loss.
        :return: Logits and loss value.
        """
        loss_val = 0
        if forecasting_model is not None and residual_model is not None:
            enc_outputs, dec_outputs = self.classifier(enc_inputs, dec_inputs)
            dec_outputs_res = residual_model.predict(enc_outputs, dec_outputs, embed=False)
            logits = self.final_projection(dec_outputs + dec_outputs_res)

        else:
            enc_outputs, dec_outputs = self.classifier(enc_inputs, dec_inputs)
            logits = self.final_projection(dec_outputs)

            logits = logits[:, -self.pred_len:, :]

        if y_true is not None:
            y_true = y_true.to(self.device, dtype=torch.long).squeeze(1)
            y_true = torch.flatten(y_true, start_dim=0)
            outputs = logits.reshape(-1, 2)
            if self.class_weights is not None:
                loss_val = nn.CrossEntropyLoss(self.class_weights)(outputs, y_true)
            else:
                loss_val = nn.CrossEntropyLoss()(outputs, y_true)

        return logits, loss_val
