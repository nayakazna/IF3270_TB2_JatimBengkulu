import numpy as np

class Conv2D:
    def __init__(self, weight, bias, stride = 1, pad = 0, activation=None) -> None:
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = pad
        self.activation = activation

    def forward(self, x):
        # (Height, Width, Channel)
        if x.ndim == 3:
            return self._forward(x)
        # (Image Number, Height, Width, Channel)
        elif x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)
        else:
            raise ValueError("Input must have shape (H,W,C) or (N,H,W,C)")

    def _forward(self, x):
        H, W, C_in = x.shape
        kH, kW, C_weight, C_out = self.weight.shape

        if C_in != C_weight:
            raise ValueError(f"Input channel {C_in} != weight channel {C_weight}")

        if self.pad > 0:
            x = np.pad(
                x,
                (
                    (self.pad, self.pad),
                    (self.pad, self.pad),
                    (0, 0),
                ),
                mode="constant",
            )

        H_pad, W_pad, _ = x.shape

        H_out = (H_pad - kH) // self.stride + 1
        W_out = (W_pad - kW) // self.stride + 1

        out = np.zeros((H_out, W_out, C_out), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW

                patch = x[h_start:h_end, w_start:w_end, :]

                for c in range(C_out):
                    out[i, j, c] = np.sum(
                        patch * self.weight[:, :, :, c]
                    ) + self.bias[c]

        if self.activation is not None:
            out = self.activation(out)

        return out
