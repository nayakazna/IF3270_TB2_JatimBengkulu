import numpy as np

from .activation import sigmoid, tanh


class LSTMCell:
    def __init__(self, kernel, recurrent_kernel, bias, activation=tanh, recurrent_activation=sigmoid):
        self.kernel = np.asarray(kernel)
        self.recurrent_kernel = np.asarray(recurrent_kernel)
        self.bias = np.asarray(bias)
        self.activation = activation
        self.recurrent_activation = recurrent_activation

        if self.kernel.ndim != 2 or self.recurrent_kernel.ndim != 2:
            raise ValueError("kernel and recurrent_kernel must be 2D arrays")
        if self.kernel.shape[1] % 4 != 0:
            raise ValueError("LSTM kernel output dimension must be 4 * units")
        if self.recurrent_kernel.shape != (self.units, 4 * self.units):
            raise ValueError("recurrent_kernel shape must be (units, 4 * units)")
        if self.bias.ndim == 1 and self.bias.shape[0] != 4 * self.units:
            raise ValueError("bias size must be 4 * units")
        if self.bias.ndim == 2 and self.bias.shape != (2, 4 * self.units):
            raise ValueError("CuDNN-style LSTM bias shape must be (2, 4 * units)")

    @classmethod
    def from_keras(cls, layer, activation=tanh, recurrent_activation=sigmoid):
        weights = layer.get_weights()
        if len(weights) != 3:
            raise ValueError("Keras LSTM layer must have kernel, recurrent_kernel, and bias")
        return cls(
            weights[0],
            weights[1],
            weights[2],
            activation=activation,
            recurrent_activation=recurrent_activation,
        )

    @property
    def units(self):
        return self.kernel.shape[1] // 4

    def _bias(self):
        if self.bias.ndim == 2:
            return np.sum(self.bias, axis=0)
        return self.bias

    def forward(self, x_t, h_prev, c_prev):
        z = np.matmul(x_t, self.kernel) + np.matmul(h_prev, self.recurrent_kernel) + self._bias()
        z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)

        i = self.recurrent_activation(z_i)
        f = self.recurrent_activation(z_f)
        c_bar = self.activation(z_c)
        o = self.recurrent_activation(z_o)

        c_t = f * c_prev + i * c_bar
        h_t = o * self.activation(c_t)
        return h_t, c_t


class LSTM:
    def __init__(
        self,
        kernel,
        recurrent_kernel,
        bias,
        activation=tanh,
        recurrent_activation=sigmoid,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
    ):
        self.cell = LSTMCell(
            kernel,
            recurrent_kernel,
            bias,
            activation=activation,
            recurrent_activation=recurrent_activation,
        )
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards

    @classmethod
    def from_keras(cls, layer, activation=tanh, recurrent_activation=sigmoid):
        weights = layer.get_weights()
        if len(weights) != 3:
            raise ValueError("Keras LSTM layer must have kernel, recurrent_kernel, and bias")
        return cls(
            weights[0],
            weights[1],
            weights[2],
            activation=activation,
            recurrent_activation=recurrent_activation,
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
                f"Input feature size {input_dim} != LSTM kernel input size {self.cell.kernel.shape[0]}"
            )

        if initial_state is None:
            h_t = np.zeros((batch_size, self.units), dtype=x.dtype)
            c_t = np.zeros((batch_size, self.units), dtype=x.dtype)
        else:
            h_t, c_t = initial_state
            h_t = np.asarray(h_t)
            c_t = np.asarray(c_t)
            if h_t.ndim == 1:
                h_t = np.broadcast_to(h_t, (batch_size, self.units)).copy()
            if c_t.ndim == 1:
                c_t = np.broadcast_to(c_t, (batch_size, self.units)).copy()

        indices = range(timesteps - 1, -1, -1) if self.go_backwards else range(timesteps)
        outputs = []
        for t in indices:
            h_t, c_t = self.cell.forward(x[:, t, :], h_t, c_t)
            outputs.append(h_t)

        if self.go_backwards:
            outputs = outputs[::-1]

        sequence = np.stack(outputs, axis=1)
        result = sequence if self.return_sequences else sequence[:, -1, :]

        if squeeze_batch:
            result = result[0]
            h_t_out = h_t[0]
            c_t_out = c_t[0]
        else:
            h_t_out = h_t
            c_t_out = c_t

        if self.return_state:
            return result, h_t_out, c_t_out
        return result
