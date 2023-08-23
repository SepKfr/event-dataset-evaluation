class TransformerConfig:
    def __init__(self, vocab_size, src_input_size, d_model, n_heads, d_k, stack_size, n_layers):
        """
        Configuration class for the Transformer model.

        :param vocab_size: Size of the vocabulary.
        :param src_input_size: Size of the input source data.
        :param d_model: Dimensionality of the model.
        :param n_heads: Number of attention heads.
        :param d_k: Dimensionality of the key vectors in attention.
        :param stack_size: Number of stacked Transformer layers.
        :param n_layers: Total number of layers in the model.
        """
        self.vocab_size = vocab_size
        self.src_input_size = src_input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.stack_size = stack_size
        self.n_layers = n_layers


