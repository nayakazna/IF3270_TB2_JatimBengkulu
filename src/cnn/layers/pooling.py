import numpy as np


class MaxPooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        if x.ndim == 3:
            return self._forward(x)
        if x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        out = np.zeros((H_out, W_out, C), dtype=x.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW

                patch = x[h_start:h_end, w_start:w_end, :]
                out[i, j, :] = np.max(patch, axis=(0, 1))

        return out


class AveragePooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        if x.ndim == 3:
            return self._forward(x)
        if x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        out = np.zeros((H_out, W_out, C), dtype=x.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW

                patch = x[h_start:h_end, w_start:w_end, :]
                out[i, j, :] = np.mean(patch, axis=(0, 1))

        return out
