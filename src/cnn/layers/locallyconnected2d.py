import numpy as np


class LocallyConnected2D:
    def __init__(self, weight, bias, kernel_size, stride=1, pad=0, activation=None):
        self.weight = weight
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.activation = activation

    def forward(self, x):
        if x.ndim == 3:
            return self._forward_batch(x[np.newaxis])[0]
        elif x.ndim == 4:
            return self._forward_batch(x)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")


    def _forward_batch(self, x):
        N, H, W, C_in = x.shape
        kH, kW = self.kernel_size

        if self.pad > 0:
            x = np.pad(x, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode="constant")

        _, H_pad, W_pad, _ = x.shape
        H_out = (H_pad - kH) // self.stride + 1
        W_out = (W_pad - kW) // self.stride + 1
        num_locations = H_out * W_out

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
        col = patches.reshape(N, num_locations, -1)

        out = np.einsum('nli,lio->nlo', col, self.weight, optimize=True)

        if self.bias.ndim == 2:
            out = out + self.bias
        else:
            out = out + self.bias

        out = out.reshape(N, H_out, W_out, -1)

        if self.activation is not None:
            out = self.activation(out)

        return out.astype(np.float32)