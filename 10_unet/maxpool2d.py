"""
MaxPool2D, 2x2 window, stride 2, from scratch. Section 2: "a 2x2 max
pooling operation with stride 2 for downsampling" -- the paper uses this
exact fixed config everywhere, no other kernel/stride ever appears, so
it's not made generic. Requires even H, W (true at every pool in this
repo's scoped architecture -- see JOURNAL.md entry 4 for the dimension
arithmetic).

Implemented via the four strided quadrant views directly rather than a
reshape-then-regroup: a naive `x.reshape(N,C,Ho,2,Wo,2)` followed by
`.reshape(N,C,Ho,Wo,4)` looks like it groups each 2x2 window together,
but it doesn't -- row-major reshape interleaves sub_h with the whole Wo
axis in between, so the "4" axis wouldn't actually hold one window's
pixels. Caught this by hand-tracing the memory layout before running it;
four explicit slices sidestep the whole trap.
"""
import numpy as np


class MaxPool2x2:
    def forward(self, x):
        N, C, H, W = x.shape
        Ho, Wo = H // 2, W // 2
        self.x_shape = x.shape
        quads = np.stack([
            x[:, :, 0::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 1::2, 1::2],
        ], axis=-1)  # (N, C, Ho, Wo, 4)
        self.argmax = quads.argmax(axis=-1)
        return quads.max(axis=-1)

    def backward(self, dout):
        dx = np.zeros(self.x_shape, dtype=dout.dtype)
        slices = [
            (slice(0, None, 2), slice(0, None, 2)),
            (slice(0, None, 2), slice(1, None, 2)),
            (slice(1, None, 2), slice(0, None, 2)),
            (slice(1, None, 2), slice(1, None, 2)),
        ]
        for k, (sh, sw) in enumerate(slices):
            dx[:, :, sh, sw] = dout * (self.argmax == k)
        return dx
