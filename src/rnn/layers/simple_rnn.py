import numpy as np

from .activation import tanh


class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias, activation=tanh):
        self.kernel = np.asarray(kernel)
        self.recurrent_kernel = np.asarray(recurrent_kernel)
        self.bias = np.asarray(bias)
        self.activation = activation

        if self.kernel.ndim != 2 or self.recurrent_kernel.ndim != 2:
            raise ValueError("kernel and recurrent_kernel must be 2D arrays")
        if self.kernel.shape[1] != self.recurrent_kernel.shape[0]:
            raise ValueError("kernel units must match recurrent kernel units")
        if self.recurrent_kernel.shape[0] != self.recurrent_kernel.shape[1]:
            raise ValueError("recurrent_kernel must be square")
        if self.bias.shape[-1] != self.units:
            raise ValueError("bias size must match units")

    @classmethod
    def from_keras(cls, layer, activation=tanh):
        weights = layer.get_weights()
        if len(weights) != 3:
            raise ValueError("Keras SimpleRNN layer must have kernel, recurrent_kernel, and bias")
        return cls(weights[0], weights[1], weights[2], activation=activation)

    @property
    def units(self):
        return self.recurrent_kernel.shape[0]

    def forward(self, x_t, h_prev):
        x_t = np.asarray(x_t)
        h_prev = np.asarray(h_prev)
        out = np.matmul(x_t, self.kernel) + np.matmul(h_prev, self.recurrent_kernel) + self.bias
        return self.activation(out)


class SimpleRNN:
    def __init__(
        self,
        kernel,
        recurrent_kernel,
        bias,
        activation=tanh,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
    ):
        self.cell = SimpleRNNCell(kernel, recurrent_kernel, bias, activation=activation)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards

    @classmethod
    def from_keras(cls, layer, activation=tanh):
        weights = layer.get_weights()
        if len(weights) != 3:
            raise ValueError("Keras SimpleRNN layer must have kernel, recurrent_kernel, and bias")
        return cls(
            weights[0],
            weights[1],
            weights[2],
            activation=activation,
            return_sequences=getattr(layer, "return_sequences", True),
            return_state=getattr(layer, "return_state", False),
            go_backwards=getattr(layer, "go_backwards", False),
        )

    @property
    def units(self):
        return self.cell.units

    def forward(self, x, initial_state=None):
        x = np.asarray(x)
        squeeze_batch = False
        if x.ndim == 2:
            x = x[np.newaxis, ...]
            squeeze_batch = True
        elif x.ndim != 3:
            raise ValueError("Input must have shape (T, D) or (N, T, D)")

        batch_size, timesteps, input_dim = x.shape
        if input_dim != self.cell.kernel.shape[0]:
            raise ValueError(
                f"Input feature size {input_dim} != RNN kernel input size {self.cell.kernel.shape[0]}"
            )

        if initial_state is None:
            h_t = np.zeros((batch_size, self.units), dtype=x.dtype)
        else:
            h_t = np.asarray(initial_state)
            if h_t.ndim == 1:
                h_t = np.broadcast_to(h_t, (batch_size, self.units)).copy()

        indices = range(timesteps - 1, -1, -1) if self.go_backwards else range(timesteps)
        outputs = []
        for t in indices:
            h_t = self.cell.forward(x[:, t, :], h_t)
            outputs.append(h_t)

        if self.go_backwards:
            outputs = outputs[::-1]

        sequence = np.stack(outputs, axis=1)
        result = sequence if self.return_sequences else sequence[:, -1, :]

        if squeeze_batch:
            result = result[0]
            h_t_out = h_t[0]
        else:
            h_t_out = h_t

        if self.return_state:
            return result, h_t_out
        return result
