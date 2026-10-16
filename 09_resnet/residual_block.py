"""
BasicBlock (paper Fig.2/Fig.5-left, the 2-layer F used for ResNet-20):
conv3x3 -> BN -> ReLU -> conv3x3 -> BN, added to the shortcut, then ReLU
(Section 3.2: "we adopt the second nonlinearity after the addition").

IdentityShortcut is option A (JOURNAL.md entry 2/3, Section 3.2 Eqn.1 and
Section 4.2): "the shortcut still performs identity mapping, with extra
zero entries padded for increasing dimensions... performed with a stride
of 2". No parameters -- a stride-2 spatial subsample (drop every other
pixel) is NOT the same operation as a conv/pool, it's just indexing, and
is only used on the pairs of stacked 3x3 layers that cross a stage
boundary (stride=2, channels double); within a stage it's a no-op.
"""
import numpy as np
from conv2d import Conv2D
from bn2d import BatchNorm2d


class IdentityShortcut:
    def __init__(self, in_channels, out_channels, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        self.x_shape = x.shape
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.out_channels > self.in_channels:
            N, C, H, W = x.shape
            zeros = np.zeros((N, self.out_channels - self.in_channels, H, W), dtype=x.dtype)
            x = np.concatenate([x, zeros], axis=1)
        return x

    def backward(self, dout):
        if self.out_channels > self.in_channels:
            dout = dout[:, :self.in_channels, :, :]
        dx = np.zeros(self.x_shape, dtype=dout.dtype)
        if self.stride > 1:
            dx[:, :, ::self.stride, ::self.stride] = dout
        else:
            dx = dout
        return dx


class BasicBlock:
    def __init__(self, in_channels, out_channels, stride, seed=None):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, seed=seed)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1,
                             seed=None if seed is None else seed + 1)
        self.bn2 = BatchNorm2d(out_channels)
        self.needs_shortcut = (stride != 1 or in_channels != out_channels)
        self.shortcut = IdentityShortcut(in_channels, out_channels, stride) if self.needs_shortcut else None

    def params(self, prefix):
        return {
            f'{prefix}.conv1.W': self.conv1.W, f'{prefix}.conv1.b': self.conv1.b,
            f'{prefix}.conv2.W': self.conv2.W, f'{prefix}.conv2.b': self.conv2.b,
            f'{prefix}.bn1.gamma': self.bn1.gamma, f'{prefix}.bn1.beta': self.bn1.beta,
            f'{prefix}.bn2.gamma': self.bn2.gamma, f'{prefix}.bn2.beta': self.bn2.beta,
        }

    def forward(self, x, train):
        identity = self.shortcut.forward(x) if self.shortcut else x

        out = self.conv1.forward(x)
        if train:
            out, self._bn1_cache = self.bn1.forward_train(out)
        else:
            out = self.bn1.forward_eval(out)
        self._relu1_mask = out > 0
        out = np.maximum(0, out)

        out = self.conv2.forward(out)
        if train:
            out, self._bn2_cache = self.bn2.forward_train(out)
        else:
            out = self.bn2.forward_eval(out)

        out = out + identity
        self._relu2_mask = out > 0
        out = np.maximum(0, out)
        return out

    def backward(self, dout, prefix):
        dout = dout * self._relu2_mask
        d_identity = dout
        d_main = dout

        d_main, dgamma2, dbeta2 = self.bn2.backward(d_main, self._bn2_cache)
        d_main, dW2, db2 = self.conv2.backward(d_main)

        d_main = d_main * self._relu1_mask
        d_main, dgamma1, dbeta1 = self.bn1.backward(d_main, self._bn1_cache)
        d_main, dW1, db1 = self.conv1.backward(d_main)

        dx = d_main + (self.shortcut.backward(d_identity) if self.shortcut else d_identity)

        grads = {
            f'{prefix}.conv1.W': dW1, f'{prefix}.conv1.b': db1,
            f'{prefix}.conv2.W': dW2, f'{prefix}.conv2.b': db2,
            f'{prefix}.bn1.gamma': dgamma1, f'{prefix}.bn1.beta': dbeta1,
            f'{prefix}.bn2.gamma': dgamma2, f'{prefix}.bn2.beta': dbeta2,
        }
        return dx, grads
