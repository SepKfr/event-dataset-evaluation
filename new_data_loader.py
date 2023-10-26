import itertools
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, TensorDataset
from pytorch_forecasting.data import TimeSeriesDataSet
from data_formatters import oil, sev_weather, us_accident


class DataLoader:
    def __init__(self, exp_name,
                 max_encoder_length,
                 pred_len,
                 target_col,
                 max_train_sample,
                 max_test_sample,
                 batch_size):

        data_formatter = {"oil": oil.OilFormatter,
                          "sev_weather": sev_weather.SevWeatherFormatter,
                          "us_accident": us_accident.USAccidentFormatter}
        class_col = {"oil": "class", "sev_weather": "SEVPROB", "us_accident": "Severity"}

        self.max_encoder_length = max_encoder_length
        self.pred_len = pred_len
        self.max_train_sample = max_train_sample
        self.max_test_sample = max_test_sample
        self.batch_size = batch_size

        seed = 1234
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        data_csv_path = "datasets/{}.csv".format(exp_name)
        data = pd.read_csv(data_csv_path, dtype={'date': str})
        data.sort_values(by=["id", "seconds_from_start"], inplace=True)
        formatter = data_formatter[exp_name](pred_len)

        data = formatter.transform_data(data)
        self.num_classes = formatter.get_num_classes()

        total_batches = int(len(data) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.8)
        valid_len = int(total_batches * batch_size * 0.1)
        test_len = int(total_batches * batch_size * 0.1)

        train = data[:train_len]
        valid = data[train_len:train_len+valid_len]
        test = data[train_len+valid_len:train_len+valid_len+test_len]

        self.train_data = pd.DataFrame(
            dict(
                value=train[target_col],
                group=train["id"],
                time_idx=np.arange(train_len),
                c=train[class_col[exp_name]]
            )
        )

        self.valid_data = pd.DataFrame(
            dict(
                value=valid[target_col],
                group=valid["id"],
                time_idx=np.arange(train_len, train_len+valid_len),
                c=valid[class_col[exp_name]]
            )
        )

        self.test_data = pd.DataFrame(
            dict(
                value=test[target_col],
                group=test["id"],
                time_idx=np.arange(train_len+valid_len, train_len+valid_len+test_len),
                c=test[class_col[exp_name]]
            )
        )

        self.train_dataset = self.get_train_dataset(self.train_data, self.max_encoder_length, self.pred_len)
        self.valid_dataset = self.get_valid_dataset(self.valid_data, self.max_encoder_length, self.pred_len)
        self.test_dataset = self.get_test_dataset(self.test_data, self.max_encoder_length, self.pred_len)

        self.train_loader,  self.train_loader2 = \
            self.create_dataloader(self.train_data, max_train_sample)
        self.valid_loader, self.valid_loader2 = \
            self.create_dataloader(self.valid_data, max_test_sample)
        self.test_loader, self.test_loader2 = \
            self.create_dataloader(self.test_data, max_test_sample)

    def get_train_dataset(self, train_data, min_encoder_length, min_prediction_length):
        return self.create_time_series_dataset(train_data, min_encoder_length, min_prediction_length)

    def get_valid_dataset(self, valid_data, min_encoder_length, min_prediction_length):
        return self.create_time_series_dataset(valid_data, min_encoder_length, min_prediction_length)

    def get_test_dataset(self, test_data, min_encoder_length, min_prediction_length):
        return self.create_time_series_dataset(test_data, min_encoder_length, min_prediction_length)

    def create_time_series_dataset(self, data, t, min_encoder_length=1, min_prediction_length=1):
        return TimeSeriesDataSet(
            data,
            group_ids=["group"],
            target="value",
            time_idx="time_idx",
            min_encoder_length=min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=min_prediction_length,
            max_prediction_length=self.pred_len,
            time_varying_unknown_reals=["value", "c"],
        ) if t == "value" else TimeSeriesDataSet(
            data,
            group_ids=["group"],
            target=["value", "c"],
            time_idx="time_idx",
            min_encoder_length=min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=min_prediction_length,
            max_prediction_length=self.pred_len,
            time_varying_unknown_reals=["value", "c"],
        )

    def create_dataloader(self, data, num_samples):
        batch_sampler = BatchSampler(
            sampler=torch.utils.data.RandomSampler(data, num_samples=num_samples),
            batch_size=self.batch_size,
            drop_last=False,
        )
        data_loader = self.create_time_series_dataset(data, "value").to_dataloader(batch_sampler=batch_sampler)
        data_loader_2 = self.create_time_series_dataset(data, "c").to_dataloader(batch_sampler=batch_sampler)
        x_enc_list = []
        x_dec_list = []
        y_list = []
        y_true_con = []
        for x, y in data_loader:
            x_enc_list.append(x["encoder_cont"][:, :-self.pred_len, :])
            x_dec_list.append(x["encoder_cont"][:, -self.pred_len:, :])
            y_list.append(y[0].unsqueeze(-1))

        for _, y in data_loader_2:
            y = torch.stack(list(itertools.chain.from_iterable(y[0])))
            y_true_con.append(y.reshape(self.batch_size, self.pred_len, -1))

        x_enc = torch.stack(list(itertools.chain.from_iterable(x_enc_list)))
        x_dec = torch.stack(list(itertools.chain.from_iterable(x_dec_list)))
        y = torch.stack(list(itertools.chain.from_iterable(y_list)))
        y_true_con = torch.stack(list(itertools.chain.from_iterable(y_true_con)))

        tensor_dataset = TensorDataset(x_enc,
                                       x_dec,
                                       y,
                                       y_true_con)

        return torch.utils.data.DataLoader(tensor_dataset, batch_size=self.batch_size), data_loader