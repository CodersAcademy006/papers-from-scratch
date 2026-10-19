"""
"Up-convolution" from scratch (Section 2): "an upsampling of the feature
map followed by a 2x2 convolution that halves the number of feature
channels." Fig.1 shows this step doubling spatial size exactly (e.g.
28x28 -> 56x56, 1024ch -> 512ch) with no cropping on the upsampled side
itself -- only the *skip* feature map gets cropped (crop_concat.py), not
this one. Kernel size (2) == stride (2), no overlap, no padding: that
makes this mathematically a true transposed convolution, not a fixed
upsample (nearest/bilinear) followed by an ordinary conv -- each single
input pixel maps, via a learned per-channel linear map, to an
independent, non-overlapping 2x2 output block. (A fixed-upsample
substitute would have no learnable weights in the upsampling step
itself, which is not what Fig.1 depicts.)

Because the four (i,j) output offsets never overlap, forward/backward
are direct strided assignment, not an accumulating scatter -- no
col2im-style "+=" needed for dx or dW beyond summing the two (i,j) loop
contributions, which touch disjoint W slices, not disjoint dW/dx.

He init: fan_in here is Cin * 1 (each output pixel is a function of
exactly one input pixel, not a 2x2 neighborhood of them), matching the
"N = number of incoming nodes of one neuron" definition in Section 3.
"""
import numpy as np


class TransposedConv2x2Stride2:
    def __init__(self, in_channels, out_channels, bias_init=0.0, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        rng = np.random.default_rng(seed)
        fan_in = in_channels
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(in_channels, out_channels, 2, 2))
        self.b = np.full(out_channels, bias_init, dtype=np.float64)

    def forward(self, x):
        N, C, H, W = x.shape
        self.x = x
        out = np.empty((N, self.out_channels, H * 2, W * 2), dtype=x.dtype)
        for i in range(2):
            for j in range(2):
                out[:, :, i::2, j::2] = np.einsum('nchw,co->nohw', x, self.W[:, :, i, j])
        out += self.b[None, :, None, None]
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        dW = np.zeros_like(self.W)
        for i in range(2):
            for j in range(2):
                dslice = dout[:, :, i::2, j::2]
                dW[:, :, i, j] = np.einsum('nohw,nchw->co', dslice, self.x)
                dx += np.einsum('nohw,co->nchw', dslice, self.W[:, :, i, j])
        db = dout.sum(axis=(0, 2, 3))
        return dx, dW, db
