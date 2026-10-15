"""
Conv2D from scratch (im2col, no framework conv op). Supports C3's
irregular per-output-map input connectivity (Table I, JOURNAL.md entry 2)
via a boolean (C_out, C_in) connection mask applied to the weight tensor
every forward pass -- unconnected (output_map, input_map) weight slices
stay exactly zero forever, since dL/dW_effective = dL/dW * mask falls out
directly from W_effective = W * mask.

valid (no padding) convolution only, stride 1 -- matches every conv layer
in LeNet-5 (C1, C3, C5).
"""
import numpy as np


def im2col(x, kh, kw):
    N, C, H, W = x.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    cols = np.empty((N, C, kh, kw, Ho, Wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[:, :, i:i + Ho, j:j + Wo]
    return cols.reshape(N, C * kh * kw, Ho * Wo)


def col2im(dcols, x_shape, kh, kw):
    N, C, H, W = x_shape
    Ho, Wo = H - kh + 1, W - kw + 1
    dcols = dcols.reshape(N, C, kh, kw, Ho, Wo)
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i + Ho, j:j + Wo] += dcols[:, :, i, j, :, :]
    return dx


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, connection_table=None, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        rng = np.random.default_rng(seed)

        if connection_table is None:
            self.mask = np.ones((out_channels, in_channels), dtype=np.float64)
        else:
            self.mask = np.zeros((out_channels, in_channels), dtype=np.float64)
            for o, ins in enumerate(connection_table):
                self.mask[o, ins] = 1.0

        fan_in_per_map = self.mask.sum(axis=1) * kernel_size * kernel_size  # F_i, Appendix A
        self.W = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for o in range(out_channels):
            bound = 2.4 / fan_in_per_map[o]
            self.W[o] = rng.uniform(-bound, bound, size=(in_channels, kernel_size, kernel_size))
        self.W *= self.mask[:, :, None, None]
        self.b = np.zeros(out_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kh, self.kw
        Ho, Wo = H - kh + 1, W - kw + 1
        self.x_shape = x.shape
        self.cols = im2col(x, kh, kw)  # (N, C*kh*kw, Ho*Wo)
        W_eff = (self.W * self.mask[:, :, None, None]).reshape(self.out_channels, -1)
        out = np.einsum('oc,ncs->nos', W_eff, self.cols) + self.b[None, :, None]
        return out.reshape(N, self.out_channels, Ho, Wo)

    def backward(self, dout):
        N, C_out, Ho, Wo = dout.shape
        dout_flat = dout.reshape(N, C_out, Ho * Wo)
        W_eff = (self.W * self.mask[:, :, None, None]).reshape(self.out_channels, -1)

        db = dout_flat.sum(axis=(0, 2))
        dW = np.einsum('nos,ncs->oc', dout_flat, self.cols)
        dW = dW.reshape(self.W.shape) * self.mask[:, :, None, None]

        dcols = np.einsum('oc,nos->ncs', W_eff, dout_flat)
        dx = col2im(dcols, self.x_shape, self.kh, self.kw)
        return dx, dW, db
