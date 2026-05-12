import numpy as np


class MaxPooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride    = stride if stride is not None else self.pool_size
        self.stride    = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        if x.ndim == 3:
            return self._forward_batch(x[np.newaxis])[0]
        elif x.ndim == 4:
            return self._forward_batch(x)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward_batch(self, x):
        N, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        shape   = (N, H_out, W_out, pH, pW, C)
        strides = (
            x.strides[0],
            x.strides[1] * sH,
            x.strides[2] * sW,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return patches.max(axis=(3, 4)).astype(x.dtype)


class AveragePooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride    = stride if stride is not None else self.pool_size
        self.stride    = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        if x.ndim == 3:
            return self._forward_batch(x[np.newaxis])[0]
        elif x.ndim == 4:
            return self._forward_batch(x)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward_batch(self, x):
        N, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        shape   = (N, H_out, W_out, pH, pW, C)
        strides = (
            x.strides[0],
            x.strides[1] * sH,
            x.strides[2] * sW,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return patches.mean(axis=(3, 4)).astype(x.dtype)