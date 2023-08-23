import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, seed, device, dropout=0.0):
        """
        Position-wise Feed Forward Neural Network in the Transformer model.

        :param d_model: Dimensionality of the model.
        :param d_ff: Dimensionality of the feedforward layer.
        :param seed: Random seed for reproducibility.
        :param device: Device on which the model is executed.
        :param dropout: Dropout rate.
        """
        super(PoswiseFeedForwardNet, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        factory_kwargs = {'device': device}
        self.linear1 = nn.Linear(d_model, d_ff, **factory_kwargs)
        self.linear2 = nn.Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x):
        x = self.linear2(F.relu(self.linear1(x)))
        return x
