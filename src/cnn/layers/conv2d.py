import numpy as np

class Conv2D:
    def __init__(self, weight, bias, stride = 1, pad = 0, activation=None) -> None:
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = pad
        self.activation = activation

    def forward(self, x):
        if x.ndim == 3:
            return self._forward_batch(x[np.newaxis])[0]
        elif x.ndim == 4:
            return self._forward_batch(x)
        else:
            raise ValueError("Input must have shape (H,W,C) or (N,H,W,C)")

    def _forward_batch(self, x):
        N, H, W, C_in = x.shape
        kH, kW, _, C_out = self.weight.shape

        if self.pad > 0:
            x = np.pad(x, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode="constant")

        _, H_pad, W_pad, _ = x.shape
        H_out = (H_pad - kH) // self.stride + 1
        W_out = (W_pad - kW) // self.stride + 1

        shape   = (N, H_out, W_out, kH, kW, C_in)
        strides = (
            x.strides[0],
            x.strides[1] * self.stride,
            x.strides[2] * self.stride,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        col = patches.reshape(N, H_out, W_out, -1)

        W_col = self.weight.reshape(-1, C_out)

        out = col @ W_col + self.bias

        if self.activation is not None:
            out = self.activation(out)

        return out.astype(np.float32)