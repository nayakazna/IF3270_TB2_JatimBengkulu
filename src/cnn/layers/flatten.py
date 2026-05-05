import numpy as np


class Flatten:
    def forward(self, x):
        if x.ndim == 3:
            return self._forward(x)
        elif x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)

        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        return x.reshape(-1)
    
