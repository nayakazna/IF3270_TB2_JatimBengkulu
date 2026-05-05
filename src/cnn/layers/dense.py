import numpy as np

class Dense:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, x):
        if x.ndim == 1:
            out = self._forward(x)
        elif x.ndim == 2:
            out = np.stack([self._forward(sample) for sample in x], axis=0)
        else:
            raise ValueError("Input must have shape (in_features,) or (N, in_features)")

        if self.activation is not None:
            out = self.activation(out)

        return out

    def _forward(self, x):
        return x @ self.weights + self.bias
