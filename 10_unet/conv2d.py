"""
Conv2D from scratch (im2col, no framework conv op). Duplicated from
09_resnet/conv2d.py (self-contained-folder convention), but padding is
dropped entirely: Section 2 of the U-Net paper is explicit that the
network "only uses the valid part of each convolution" -- no padding is
ever used anywhere in this architecture, by design (it's what makes the
overlap-tile seamless-inference strategy work). Keeping a dead `padding`
param around to match prior folders would misrepresent that as an
unused generality rather than a real constraint of this paper.
Used here with kernel_size=3 (every double-conv block) and kernel_size=1
(the final classifier conv), stride always 1.

He init (std=sqrt(2/fan_in)) -- Section 3 specifies this exact formula
for this architecture ("a Gaussian distribution with a standard
deviation of sqrt(2/N)").
"""
import numpy as np


def im2col(x, kh, kw):
    N, C, H, W = x.shape
    Ho = H - kh + 1
    Wo = W - kw + 1
    cols = np.empty((N, C, kh, kw, Ho, Wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[:, :, i:i + Ho, j:j + Wo]
    return cols.reshape(N, C * kh * kw, Ho * Wo), Ho, Wo


def col2im(dcols, x_shape, kh, kw):
    N, C, H, W = x_shape
    Ho = H - kh + 1
    Wo = W - kw + 1
    dcols = dcols.reshape(N, C, kh, kw, Ho, Wo)
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i + Ho, j:j + Wo] += dcols[:, :, i, j, :, :]
    return dx


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, bias_init=0.0, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.full(out_channels, bias_init, dtype=np.float64)

    def forward(self, x):
        self.x_shape = x.shape
        self.cols, Ho, Wo = im2col(x, self.kh, self.kw)
        W_flat = self.W.reshape(self.out_channels, -1)
        out = np.einsum('oc,ncs->nos', W_flat, self.cols) + self.b[None, :, None]
        return out.reshape(x.shape[0], self.out_channels, Ho, Wo)

    def backward(self, dout):
        N, C_out, Ho, Wo = dout.shape
        dout_flat = dout.reshape(N, C_out, Ho * Wo)
        W_flat = self.W.reshape(self.out_channels, -1)

        db = dout_flat.sum(axis=(0, 2))
        dW = np.einsum('nos,ncs->oc', dout_flat, self.cols).reshape(self.W.shape)

        dcols = np.einsum('oc,nos->ncs', W_flat, dout_flat)
        dx = col2im(dcols, self.x_shape, self.kh, self.kw)
        return dx, dW, db
