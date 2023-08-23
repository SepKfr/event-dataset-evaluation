import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os.path
import random
from sklearn.metrics import accuracy_score, f1_score

'''
This file creates the sub figures in Figure 3 of the manuscript.
'''


def create_synthetic_continuous_figure(true_value, predictions, residual_predictions):
    np.random.seed(89)
    # Generate time values
    time_input = np.arange(0, 50.1, 0.1)
    time_prediction = np.arange(50, 75.1, 0.1)

    # Generate data values with a smooth pattern
    data_input = 5 + 5 * np.sin(2 * np.pi * time_input / 50)
    data_true = 5 + 5 * np.sin(2 * np.pi * time_prediction / 50)  # True values after 50

    data_input = (data_input - np.mean(data_input)) / np.std(data_input)
    data_true = (data_true - np.mean(data_true)) / np.std(data_true)

    data_prediction = data_true + np.random.normal(scale=0.9, size=len(time_prediction)) # Predictions with added error

    # Generate a random weight matrix for embedding
    embedding_weight = np.random.normal(size=(len(data_input), len(data_input)))
    embedding_weight_y_true = np.random.normal(size=(len(data_true), len(data_true)))

    # Apply embedding to the centered input data
    data_input_embedded = np.dot(embedding_weight, data_input).flatten()
    data_true_embedded = np.dot(embedding_weight_y_true, data_true).flatten()
    pred_embedded = np.dot(embedding_weight_y_true, data_prediction).flatten()

    # Apply Savitzky-Golay filter for noise reduction
    smoothed_data_input_embedded = savgol_filter(data_input_embedded, window_length=80, polyorder=5)
    smoothed_data_true_embedded = savgol_filter(data_true_embedded, window_length=80, polyorder=5)
    smoothed_pred_embedded = savgol_filter(pred_embedded, window_length=80, polyorder=5)

    smoothed_data_true_embedded = np.hstack([smoothed_data_input_embedded[-1], smoothed_data_true_embedded[1:]])
    smoothed_pred_embedded = np.hstack([smoothed_data_input_embedded[-1], smoothed_pred_embedded[1:]])

    res =(smoothed_data_true_embedded - smoothed_pred_embedded) + 35 *np.random.normal(scale=0.1, size=len(time_prediction))
    res = savgol_filter(res, window_length=80, polyorder=5)
    res_pred = res + smoothed_pred_embedded

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.rcParams["legend.fontsize"] = 24

    plt.plot(time_input, smoothed_data_input_embedded, linewidth=2, color="grey", alpha=0.8)

    if true_value:
        plt.plot(time_prediction, smoothed_data_true_embedded, label='True Values', linewidth=2, color="grey", alpha=0.8)
        name_of_figure = "true_values"
    elif predictions:
        plt.plot(time_prediction, smoothed_data_true_embedded, label='True Values', linewidth=2, color="grey",
                 alpha=0.8)
        plt.plot(time_prediction, smoothed_pred_embedded, label='Predictions', linewidth=2, color="darkblue")
        name_of_figure = "only_predictions"
    elif residual_predictions:
        plt.plot(time_prediction, smoothed_data_true_embedded, label='True Values', linewidth=2, color="grey",
                 alpha=0.8)
        plt.plot(time_prediction, res_pred, label='Boosted Predictions', linewidth=2, color="darkgreen")
        name_of_figure = "residual_predictions"
    else:
        plt.plot(time_prediction, smoothed_data_true_embedded, label='True Values', linewidth=2, color="grey",
                 alpha=0.8)
        plt.plot(time_prediction, res, label='Residuals', linewidth=2, color="tomato")
        name_of_figure = "residuals"

    # Add a vertical line to separate input and prediction periods
    plt.axvline(x=50, color='gray', linestyle='dotted', linewidth=2)

    # Set labels and title
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Set x-axis limits and ticks

    # Set legend
    plt.legend()

    # Show the plot
    plt.tight_layout()

    if not os.path.exists("Images_for_Fig_3"):
        os.makedirs("Images_for_Fig_3")
    plt.savefig("Images_for_Fig_3/{}.png".format(name_of_figure), dpi=500)
    plt.show()


def create_synthetic_discrete(ground_truth, only_pred):

    np.random.seed(1235)
    random.seed(1235)
    # Generate time values
    time_input = np.arange(0, 50.1, 1)
    time_prediction = np.arange(50, 75.1, 1)

    # Generate a base pattern
    base_pattern = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    base_pattern_input = np.tile(base_pattern, len(time_input) // len(base_pattern) + 1)[:len(time_input)]
    base_pattern_prediction = np.tile(base_pattern, len(time_prediction) // len(base_pattern) + 1)[:len(time_prediction)]

    # Add more random variations
    random_variation_input = np.random.randint(-4, 2, size=len(time_input))
    random_variation_prediction = np.random.randint(-4, 2, size=len(time_prediction))
    if only_pred:
        random_variation_prediction_2 = np.random.randint(-4, 2, size=len(time_prediction))
        name = "predictions_discrete_by_standalone"
    else:
        random_variation_prediction_2 = np.random.randint(-1, 3, size=len(time_prediction))
        name = "predictions_discrete_by_res_pred"
    # Generate data values with base pattern and increased random noise
    data_input = base_pattern_input + random_variation_input
    data_true = base_pattern_prediction + random_variation_prediction
    data_preds = data_true + random_variation_prediction_2

    # Set values less than 0 to 0
    data_input[data_input < 0] = np.random.randint(0, 4)
    data_true[data_true < 0] = np.random.randint(0, 4)
    data_preds[data_preds < 0] = np.random.randint(0, 4)

    data_input[data_input > 10] = np.random.randint(0, 10)
    data_true[data_true > 10] = np.random.randint(0, 10)
    data_preds[data_preds > 10] = np.random.randint(0, 10)

    # Set abnormal values to red
    abnormal_indices_input = np.where(data_input > 4)
    abnormal_indices_prediction = np.where(data_true > 4)
    colors_input = ['red' if i in abnormal_indices_input[0] else 'blue' for i in range(len(data_input))]
    colors_prediction = ['red' if i in abnormal_indices_prediction[0] else 'blue' for i in range(len(data_true))]

    # Create the plot
    plt.figure(figsize=(10, 6))

    y_true = np.where(data_true >= 5, 1, 0)
    y_pred = np.where(data_preds >= 5, 1, 0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    plt.scatter(time_input, data_input, c=colors_input, marker='o')
    if ground_truth:
        plt.scatter(time_prediction, data_true, c=colors_prediction, marker='o')
        name = "true_values_discrete"
    else:
        plt.scatter(time_prediction, data_preds, c=colors_prediction)
        title = plt.title('Accuracy = {:.3f} \nF1-score = {:.3f}'.format(acc, f1))
        title.set_fontsize(28)

    plt.ylim(-1, 11)

    # Add a dotted line to separate input and prediction periods
    plt.axvline(x=50, color='gray', linestyle='dotted')

    # Set labels and title
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Events', fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.tight_layout()
    if not os.path.exists("Images_for_Fig_3"):
        os.makedirs("Images_for_Fig_3")

    plt.savefig("Images_for_Fig_3/{}.png".format(name), dpi=500)
    plt.show()


def main():

    # only create continuous data with true values
    create_synthetic_continuous_figure(true_value=True, predictions=False, residual_predictions=False)
    # create continuous data with predictions made by standalone classification model
    create_synthetic_continuous_figure(true_value=False, predictions=True, residual_predictions=False)
    # create continuous data with augmented predictions with residuals
    create_synthetic_continuous_figure(true_value=False, predictions=False, residual_predictions=True)
    # create continuous data with only residuals
    create_synthetic_continuous_figure(true_value=False, predictions=False, residual_predictions=False)
    # create synthetic discrete data for only ground_truth
    create_synthetic_discrete(ground_truth=True, only_pred=False)
    # create synthetic discrete data for predictions made by standalone classification model
    create_synthetic_discrete(ground_truth=False, only_pred=False)
    # create synthetic discrete data with augmented predictions with residuals
    create_synthetic_discrete(ground_truth=False, only_pred=True)


if __name__ == '__main__':
    main()



