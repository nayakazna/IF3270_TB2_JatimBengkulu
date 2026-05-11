import numpy as np


class Embedding:
    def __init__(self, weights, mask_zero=False):
        self.weights = np.asarray(weights)
        self.mask_zero = mask_zero

    @classmethod
    def from_keras(cls, layer):
        weights = layer.get_weights()
        if len(weights) != 1:
            raise ValueError("Keras Embedding layer must have exactly one weight matrix")
        return cls(weights[0], mask_zero=getattr(layer, "mask_zero", False))

    @property
    def input_dim(self):
        return self.weights.shape[0]

    @property
    def output_dim(self):
        return self.weights.shape[1]

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids)
        if not np.issubdtype(token_ids.dtype, np.integer):
            token_ids = token_ids.astype(np.int64)

        if np.any(token_ids < 0) or np.any(token_ids >= self.input_dim):
            raise ValueError("Token ids contain values outside the embedding vocabulary")

        return self.weights[token_ids]
