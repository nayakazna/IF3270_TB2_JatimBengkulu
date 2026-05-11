import numpy as np


class Dense:
    def __init__(self, weights, bias, activation=None):
        self.weights = np.asarray(weights)
        self.bias = np.asarray(bias)
        self.activation = activation

    @classmethod
    def from_keras(cls, layer, activation=None):
        weights = layer.get_weights()
        if len(weights) != 2:
            raise ValueError("Keras Dense layer must have kernel and bias weights")
        return cls(weights[0], weights[1], activation=activation)

    def forward(self, x):
        x = np.asarray(x)
        if x.shape[-1] != self.weights.shape[0]:
            raise ValueError(
                f"Input feature size {x.shape[-1]} != weight input size {self.weights.shape[0]}"
            )

        out = np.matmul(x, self.weights) + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out
