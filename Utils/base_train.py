# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Link to repository: https://github.com/google-research/google-research/tree/master/tft
import pandas as pd
# Lint as: python3

import torch
import numpy as np
from Utils import utils, base
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

InputTypes = base.InputTypes


def sample_train_val_test(ddf, max_samples, time_steps, num_encoder_steps, pred_len, column_definition):
    """
    Sample training, validation, and testing data from the input dataframe.

    :param ddf: The input dataframe.
    :param max_samples: Maximum number of samples to generate.
    :param time_steps: Total time steps in each sample.
    :param num_encoder_steps: Number of time steps assigned to the encoder.
    :param pred_len: Length of the prediction horizon.
    :param column_definition: Definition of each column in the dataset.
    :return: Dictionary containing sampled data arrays.
    """

    # Extract column names based on input types
    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)
    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definition)
    enc_input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]
    class_col = utils.get_single_col_by_input_type(InputTypes.KNOWN_INPUT, column_definition)

    valid_sampling_locations = []
    split_data_map = {}

    # Group input data by identifier and create valid sampling locations
    for identifier, df in ddf.groupby(id_col):
        num_entries = len(df)
        if num_entries >= time_steps:
            valid_sampling_locations += [
                (identifier, time_steps + i)
                for i in range(num_entries - time_steps + 1)
            ]
        split_data_map[identifier] = df

    # Randomly choose valid sampling locations for generating samples
    ranges = [valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)]

    input_size = len(enc_input_cols)
    inputs = np.zeros((max_samples, time_steps, input_size))
    enc_inputs = np.zeros((max_samples, num_encoder_steps, input_size))
    dec_inputs = np.zeros((max_samples, time_steps - num_encoder_steps - pred_len, input_size))
    outputs = np.zeros((max_samples, pred_len, 1))
    outputs_forecasting = np.zeros((max_samples, pred_len, 2))
    time = np.empty((max_samples, time_steps, 1), dtype=object)
    identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

    start = 0
    # Iterate over ranges to generate samples
    for i, tup in enumerate(ranges):
        if (i + 1 % 1000) == 0:
            print(i + 1, 'of', max_samples, 'samples done...')
        ident, start_idx = tup
        sliced = split_data_map[ident].iloc[start_idx - time_steps:start_idx]
        enc_inputs[i + start, :, :] = sliced[enc_input_cols].iloc[:num_encoder_steps]
        dec_inputs[i + start, :, :] = sliced[enc_input_cols].iloc[num_encoder_steps:-pred_len]
        inputs[i + start, :, :] = sliced[enc_input_cols]
        outputs[i + start, :, :] = sliced[[target_col]].iloc[-pred_len:]
        outputs_forecasting[i + start, :, :] = sliced[[class_col, target_col]].iloc[-pred_len:]
        time[i + start, :, 0] = sliced[time_col]
        identifiers[i + start, :, 0] = sliced[id_col]

    # Create a dictionary containing sampled data arrays
    sampled_data = {
        'inputs': inputs[:, :-pred_len, :],
        'enc_inputs': enc_inputs,
        'dec_inputs': dec_inputs,
        'outputs': outputs,
        'forecasting_output': outputs_forecasting,
    }

    return sampled_data


def batch_sampled_data(data, train_percent, max_samples, time_steps,
                       num_encoder_steps, pred_len,
                       column_definition, batch_size):
    """
    Batch and sample segments of data into compatible formats for training, validation, and testing.

    :param data: Source dataset to sample and batch.
    :param train_percent: Percentage of data to use for training.
    :param max_samples: Maximum number of samples in each batch.
    :param time_steps: Total time steps in each sample.
    :param num_encoder_steps: Number of time steps assigned to the encoder.
    :param pred_len: Length of the prediction horizon.
    :param column_definition: Definition of each column in the dataset.
    :param batch_size: Batch size for training, validation, and testing data loaders.
    :return: Train, validation, and test data loaders.
    """

    # Set random seeds for reproducibility
    np.random.seed(1234)
    random.seed(1234)

    # Extract necessary column names based on input types
    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)

    # Sort data based on ID and time columns
    data.sort_values(by=[id_col, time_col], inplace=True)

    # Split data into training, validation, and testing sets
    train_len = int(len(data) * train_percent)
    valid_len = int((len(data) - train_len) / 2)
    train = data[:train_len]
    valid = data[train_len:-valid_len]
    test = data[-valid_len:]

    # Extract maximum samples for training and validation
    train_max, valid_max = max_samples

    # Sample data for training, validation, and testing
    sample_train = sample_train_val_test(train, train_max, time_steps, num_encoder_steps, pred_len, column_definition)
    sample_valid = sample_train_val_test(valid, valid_max, time_steps, num_encoder_steps, pred_len, column_definition)
    sample_test = sample_train_val_test(test, valid_max, time_steps, num_encoder_steps, pred_len, column_definition)

    # Convert forecasting output to torch.FloatTensor
    y_true_for_train = torch.FloatTensor(sample_train['forecasting_output'])
    y_true_for_valid = torch.FloatTensor(sample_valid['forecasting_output'])
    y_true_for_test = torch.FloatTensor(sample_test['forecasting_output'])

    def convert_continuous(y_true):
        """
        Convert categorical target values to continuous form.

        :param y_true: Target values to be converted.
        :return: continuous target values.
        """
        y_true_2d = y_true.reshape(-1, 3)
        y_true_cont = pd.DataFrame(y_true_2d).ewm(alpha=2/3).mean()
        y_true_cont = y_true_cont.to_numpy()
        y_true_cont = torch.FloatTensor(y_true_cont)
        y_true_cont = y_true_cont.reshape(y_true.shape)
        return y_true_cont

    # Create datasets for training, validation, and testing
    train_data = TensorDataset(torch.FloatTensor(sample_train['enc_inputs']),
                               torch.FloatTensor(sample_train['dec_inputs']),
                               torch.FloatTensor(sample_train['outputs']),
                               convert_continuous(y_true_for_train))

    valid_data = TensorDataset(torch.FloatTensor(sample_valid['enc_inputs']),
                               torch.FloatTensor(sample_valid['dec_inputs']),
                               torch.FloatTensor(sample_valid['outputs']),
                               convert_continuous(y_true_for_valid))

    test_data = TensorDataset(torch.FloatTensor(sample_test['enc_inputs']),
                              torch.FloatTensor(sample_test['dec_inputs']),
                              torch.FloatTensor(sample_test['outputs']),
                              convert_continuous(y_true_for_test))

    # Create data loaders for training, validation, and testing
    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, valid_data, test_data

