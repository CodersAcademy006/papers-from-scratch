"""
Conv2D from scratch (im2col, no framework conv op). Duplicated from
08_vggnet/conv2d.py (self-contained-folder convention). ResNet-20/CIFAR
uses this with stride 1 (all 3x3 convs within a stage) and stride 2
(the first conv of stages 2/3, for downsampling) -- the stride param
08_vggnet never exercised beyond 1 is exactly why it was kept generic.

He init (std=sqrt(2/fan_in)) -- for this paper it's not a substitution,
He et al. [13] is literally the init the paper cites for ResNet itself
(JOURNAL.md entry 2).
"""
import numpy as np


def pad(x, p):
    if p == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))


def im2col(x, kh, kw, stride):
    N, C, H, W = x.shape
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, Ho, Wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride]
    return cols.reshape(N, C * kh * kw, Ho * Wo), Ho, Wo


def col2im(dcols, x_shape, kh, kw, stride):
    N, C, H, W = x_shape
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    dcols = dcols.reshape(N, C, kh, kw, Ho, Wo)
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride] += dcols[:, :, i, j, :, :]
    return dx


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias_init=0.0, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.padding = padding
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.full(out_channels, bias_init, dtype=np.float64)

    def forward(self, x):
        xp = pad(x, self.padding)
        self.x_shape = x.shape
        self.xp_shape = xp.shape
        self.cols, Ho, Wo = im2col(xp, self.kh, self.kw, self.stride)
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
        dxp = col2im(dcols, self.xp_shape, self.kh, self.kw, self.stride)
        p = self.padding
        dx = dxp[:, :, p:p + self.x_shape[2], p:p + self.x_shape[3]] if p > 0 else dxp
        return dx, dW, db
