import random
import numpy as np
import torch
from torch import nn
from torch.nn import Linear
from classification_models.autoformer import AutoCorrelation


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, device, seed):
        """
        Custom Multi-Head Attention layer in the Transformer model.

        :param d_model: Dimensionality of the model.
        :param n_heads: Number of attention heads.
        :param device: Device on which the model is executed.
        :param seed: Random seed for reproducibility.
        """
        super(CustomMultiHeadAttention, self).__init__()
        factory_kwargs = {'device': device}

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        assert d_model % n_heads == 0
        d_k = d_model // n_heads

        self.WQ = Linear(d_model, d_k * n_heads, **factory_kwargs)
        self.WK = Linear(d_model, d_k * n_heads, **factory_kwargs)
        self.WV = Linear(d_model, d_k * n_heads, **factory_kwargs)
        self.fc = Linear(n_heads * d_k, d_model, **factory_kwargs)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.seed = seed

    def forward(self, Q, K, V):
        """
        Forward pass of the Custom Multi-Head Attention layer.

        :param Q: Query tensor.
        :param K: Key tensor.
        :param V: Value tensor.
        :return: Output of the multi-head attention layer.
        """
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        context, attn = AutoCorrelation(seed=self.seed)(q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2))

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.fc(context)
        return output
