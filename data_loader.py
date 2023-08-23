# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
# Link to repository: https://github.com/google-research/google-research/tree/master/tft

import argparse
import os
import numpy as np
import pandas as pd
from data_formatters.oil import OilFormatter
from data_formatters.sev_weather import SevWeatherFormatter
from data_formatters.us_accident import USAccidentFormatter


class ExperimentConfig(object):
    default_experiments = ["us_accident", "sev_weather", "oil"]

    def __init__(self, pred_len=60, experiment='oil'):
        """
        Initialize an ExperimentConfig object.

        Args:
            pred_len (int): Prediction length for the experiment.
            experiment (str): Name of the experiment.

        Raises:
            ValueError: If the specified experiment is not recognized.
        """

        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

        self.data_folder = "datasets"
        self.pred_len = pred_len
        self.experiment = experiment

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    @property
    def data_csv_path(self):
        """
        Get the path to the CSV data file for the experiment.

        Returns:
            str: Full path to the data CSV file.
        """

        csv_map = {
            "us_accident": "us_accident.csv",
            "sev_weather": "sev_weather.csv",
            "oil": "oil.csv"
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    def make_data_formatter(self):
        """
        Create a data formatter object specific to the experiment.

        Returns:
            DataFormatter: A data formatter object tailored to the experiment.
        """

        data_formatter_class = {
            "oil": OilFormatter,
            "sev_weather": SevWeatherFormatter,
            "us_accident": USAccidentFormatter
        }

        return data_formatter_class[self.experiment](self.pred_len)


def process_us_accident(exp_config, data_directory):
    """
    Process US accident data and save it to a CSV file.

    Args:
        exp_config (ExperimentConfig): Experiment configuration object.
        data_directory (str): Path to the directory containing the data.

    Returns:
        None
    """

    df = pd.read_csv(data_directory)
    df = df.dropna()
    df.index = pd.to_datetime(df["Start_Time"])
    df["id"] = df["City"]
    df["categorical_id"] = df["City"]
    df["target"] = np.where(df["Severity"] == 4, 1, 0)
    cols = ["Start_Time", "id", "categorical_id", "target", "Severity", "seconds_from_start"]

    selected_rows = df

    selected_rows.drop_duplicates()
    selected_rows.index = pd.to_datetime(selected_rows["Start_Time"])
    selected_rows.sort_index(inplace=True)

    earliest_time = selected_rows.index.min()
    selected_rows['seconds_from_start'] = (selected_rows.index - earliest_time).seconds

    selected_rows[cols].to_csv(os.path.join(exp_config.data_folder, "us_accident.csv"))


def process_sever_weather(exp_config, data_directory):
    """
    Process severe weather data and save it to a CSV file.

    Args:
        exp_config (ExperimentConfig): Experiment configuration object.
        data_directory (str): Path to the directory containing the data.

    Returns:
        None
    """

    df = pd.read_csv(data_directory)
    df = df[~(df['SEVPROB'] == -999)]
    datetime_obj = pd.to_datetime(df["X.ZTIME"], format="%Y%m%d%H%M%S")
    df.index = datetime_obj
    df["id"] = df["WSR_ID"]
    df["categorical_id"] = df["WSR_ID"]
    df["target"] = np.where(df["SEVPROB"] > 50, 1, 0)

    selected_rows = df

    selected_rows.drop_duplicates()
    selected_rows.index = pd.to_datetime(selected_rows["X.ZTIME"])
    selected_rows.sort_index(inplace=True)

    earliest_time = selected_rows.index.min()
    selected_rows['seconds_from_start'] = (selected_rows.index - earliest_time).seconds
    selected_rows.to_csv(os.path.join(exp_config.data_folder, "sev_weather.csv"))


def process_oil_event(exp_config, data_directory):
    """
    Process oil event data and save it to a CSV file.

    Args:
        exp_config (ExperimentConfig): Experiment configuration object.
        data_directory (str): Path to the directory containing the data.

    Returns:
        None
    """

    sub_dir = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    data_list = []

    # Loop through subdirectories
    for direc in sub_dir:
        data_list_in = []
        data_path_in = os.path.join(data_directory, direc)

        # Loop through files in subdirectory
        for file in os.listdir(data_path_in):
            file_path_inn = os.path.join(data_path_in, file)
            df = pd.read_csv(file_path_inn, sep=",")
            df["class"] = np.where(df["class"] > 0, int(direc), 0)
            data_list_in.append(df)

        df_dir = pd.concat(data_list_in)
        data_list.append(df_dir)

    df_final = pd.concat(data_list)
    df_final.dropna(axis=1, inplace=True)
    df_final["id"] = 1
    df_final["categorical_id"] = 1

    df_final["target"] = np.where(df_final["class"].isin([0, 1, 2, 3, 4, 5]), 0, 1)

    df_final = df_final.dropna()

    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"])

    start_time = pd.to_datetime('2017-08-09 01:00:00.000000')

    df_final = df_final[df_final["timestamp"] >= start_time]

    selected_rows = df_final

    selected_rows.index = pd.to_datetime(selected_rows["timestamp"])
    selected_rows.sort_index(inplace=True)

    earliest_time = selected_rows.index.min()

    selected_rows['seconds_from_start'] = (selected_rows.index - earliest_time).seconds
    selected_rows.to_csv(os.path.join(exp_config.data_folder, "oil.csv"))


def main(expt_name, data_directory):

    print('#### Running download script ###')
    expt_config = ExperimentConfig(experiment=expt_name)

    # Default download functions
    pre_process_functions = {
        "us_accident": process_us_accident,
        "oil": process_oil_event,
        "sev_weather": process_sever_weather
    }

    if expt_name not in pre_process_functions:
        raise ValueError('Unrecongised experiment! name={}'.format(expt_name))
    pre_process_function = pre_process_functions[expt_name]

    # Run data_set download
    print('Processing {} data_set...'.format(expt_name))
    pre_process_function(expt_config, data_directory)

    print('Process completed.')


if __name__ == '__main__':
    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument('--expt_name', type=str)
        parser.add_argument('--data_path', type=str)

        args = parser.parse_args()

        return args.expt_name, args.data_path


    name, data_path = get_args()
    main(expt_name=name, data_directory=data_path)