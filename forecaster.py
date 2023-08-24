import random
import numpy as np
import torch
import torch.nn as nn
from modules.transformer import Transformer


class Forecaster(nn.Module):
    def __init__(self, *, residual, seed, device, pred_len, config):
        """
        Forecaster module for time series forecasting.
        :param residual: Whether to use residual forecasting.
        :param seed: Random seed for reproducibility.
        :param device: Device on which the forecaster is executed.
        :param pred_len: Length of the prediction.
        :param config: Configuration parameters for the forecasting model.
        """
        super(Forecaster, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.pred_len = pred_len
        self.residual = residual

        self.forecaster = Transformer(vocab_size=config.vocab_size,
                                      src_input_size=config.src_input_size,
                                      pred_len=pred_len,
                                      device=device,
                                      seed=seed,
                                      d_model=config.d_model,
                                      n_layers=config.n_layers,
                                      n_heads=config.n_heads)

        self.proj_down_dec = nn.Linear(config.d_model*config.src_input_size, 2)

    def forward(self, enc_inputs, dec_inputs):
        """
        Forward pass of the forecaster.

        :param enc_inputs: Encoder inputs.
        :param dec_inputs: Decoder inputs.
        :param forecaster_model: Pre-trained forecasting model.
        :param y_true: True labels for calculating loss.
        :return: Loss value.
        """
        if self.residual:
            enc_outputs_res, dec_outputs_res = self.forecaster(enc_inputs, dec_inputs)
            outputs = self.proj_down_dec(dec_outputs_res)

        else:
            enc_outputs, dec_outputs = self.forecaster(enc_inputs, dec_inputs)
            outputs = self.proj_down_dec(dec_outputs)

        return outputs

    def predict(self, enc_inputs, dec_inputs, embed=True):
        """
        Perform forecasting prediction.

        :param enc_inputs: Encoder inputs.
        :param dec_inputs: Decoder inputs.
        :param embed: Whether to embed the inputs.
        :return: Forecasting outputs.
        """
        _, outputs = self.forecaster(enc_inputs, dec_inputs, embed)
        return outputs
