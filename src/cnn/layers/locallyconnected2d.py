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
            return self._forward(x)

        if x.ndim == 4:
            return np.stack([self._forward(img) for img in x], axis=0)

        raise ValueError("Input must have shape (H, W, C) or (N, H, W, C)")

    def _forward(self, x):
        H, W, C_in = x.shape
        kH, kW = self.kernel_size

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

        num_locations = H_out * W_out
        patch_size = kH * kW * C_in

        if self.weight.shape[0] != num_locations:
            raise ValueError(
                f"Weight locations {self.weight.shape[0]} != output locations {num_locations}"
            )

        if self.weight.shape[1] != patch_size:
            raise ValueError(
                f"Weight patch size {self.weight.shape[1]} != expected patch size {patch_size}"
            )

        C_out = self.weight.shape[2]
        out = np.zeros((H_out, W_out, C_out), dtype=np.float32)

        location_idx = 0

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW

                patch = x[h_start:h_end, w_start:w_end, :]
                patch_flat = patch.reshape(-1)

                local_weight = self.weight[location_idx]

                if self.bias.ndim == 2:
                    local_bias = self.bias[location_idx]
                else:
                    local_bias = self.bias

                out[i, j, :] = patch_flat @ local_weight + local_bias

                location_idx += 1

        if self.activation is not None:
            out = self.activation(out)

        return out
