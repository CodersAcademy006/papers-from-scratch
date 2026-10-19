"""
"The repeated application of two 3x3 convolutions ... each followed by a
rectified linear unit (ReLU)" (Section 2) -- the repeated building block
on both the contracting and expansive paths (and the bottleneck, which is
just one more application of this same block). ReLU implemented inline
via a forward mask, same convention as every other paper in this repo
(e.g. 09_resnet/residual_block.py) rather than a separate layer class.
"""
from conv2d import Conv2D


class DoubleConv3x3ReLU:
    def __init__(self, in_channels, mid_channels, out_channels, seed=None):
        self.conv1 = Conv2D(in_channels, mid_channels, 3, seed=seed)
        self.conv2 = Conv2D(mid_channels, out_channels, 3, seed=None if seed is None else seed + 1)

    def forward(self, x):
        a = self.conv1.forward(x)
        self._mask1 = a > 0
        a = a * self._mask1
        b = self.conv2.forward(a)
        self._mask2 = b > 0
        return b * self._mask2

    def backward(self, dout):
        dout = dout * self._mask2
        da, dW2, db2 = self.conv2.backward(dout)
        da = da * self._mask1
        dx, dW1, db1 = self.conv1.backward(da)
        grads = {'conv1.W': dW1, 'conv1.b': db1, 'conv2.W': dW2, 'conv2.b': db2}
        return dx, grads

    def params(self):
        return {'conv1.W': self.conv1.W, 'conv1.b': self.conv1.b,
                'conv2.W': self.conv2.W, 'conv2.b': self.conv2.b}
