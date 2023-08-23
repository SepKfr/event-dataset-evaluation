import torch.nn as nn
import torch
import random
import numpy as np
from modules.encoder import Encoder, EncoderLayer
from modules.decoder import Decoder, DecoderLayer
from modules.pos_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, vocab_size, src_input_size, pred_len, d_model,
                 n_heads, n_layers, device, seed):
        """
        Implementation of the Transformer model.

        :param vocab_size: Size of the vocabulary.
        :param src_input_size: Dimensionality of the source input.
        :param pred_len: Length of the prediction.
        :param d_model: Dimensionality of the model.
        :param n_heads: Number of attention heads.
        :param n_layers: Number of layers.
        :param device: Device on which the model is executed.
        :param seed: Random seed for reproducibility.
        """
        super(Transformer, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.encoder = Encoder(EncoderLayer(d_ff=d_model*4*src_input_size,
                                            d_model=d_model*src_input_size,
                                            n_heads=n_heads,
                                            device=device,
                                            seed=seed),
                               n_layers,
                               seed)
        self.decoder = Decoder(DecoderLayer(d_ff=d_model*4*src_input_size,
                                            d_model=d_model*src_input_size,
                                            n_heads=n_heads,
                                            device=device,
                                            seed=seed),
                               n_layers,
                               seed)

        self.enc_embedding = nn.Embedding(vocab_size, d_model)
        self.dec_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_emb = PositionalEncoding(d_model*src_input_size, device)
        self.pred_len = pred_len
        self.device = device

    def forward(self, enc_inputs, dec_inputs, embed=True):
        """
        Forward pass of the Transformer model.

        :param enc_inputs: Encoder inputs.
        :param dec_inputs: Decoder inputs.
        :param embed: Whether to apply embedding or not.
        :return: Encoder outputs and Decoder outputs.
        """
        if embed:
            enc_inputs = enc_inputs.to(torch.long)
            dec_inputs = dec_inputs.to(torch.long)

            enc_inputs = torch.flatten(self.enc_embedding(enc_inputs), start_dim=2)
            dec_inputs = torch.flatten(self.dec_embedding(dec_inputs), start_dim=2)

        enc_inputs = self.pos_emb(enc_inputs)
        dec_inputs = self.pos_emb(dec_inputs)

        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)

        return enc_outputs, dec_outputs
