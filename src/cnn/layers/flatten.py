import numpy as np


class Flatten:
    def forward(self, x):
        if x.ndim == 3:
            return x.reshape(-1)
        elif x.ndim == 4:
            return x.reshape(x.shape[0], -1)
        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")
    
