# src/cnn/layers/global_pooling.py

import numpy as np


class GlobalMaxPooling2D:
    def forward(self, x):
        if x.ndim == 3:
            return self._forward(x)

        if x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)

        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        return np.max(x, axis=(0, 1))


class GlobalAveragePooling2D:
    def forward(self, x): 
        if x.ndim == 3:
            return self._forward(x)

        if x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)

        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        return np.mean(x, axis=(0, 1))
