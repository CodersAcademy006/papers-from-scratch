"""
Skip connection: center-crop the contracting-path feature map down to the
upsampled feature map's spatial size, then concatenate along the channel
axis. Section 2: "a concatenation with the correspondingly cropped
feature map from the contracting path. The cropping is necessary due to
the loss of border pixels in every convolution." Center crop (not a
corner crop) because every valid conv/pool/up-conv in this network is
spatially symmetric, so the surviving region after any number of them is
always centered in the original input -- a corner crop would silently
misalign the skip features with what the upsampled path is actually
looking at.
"""
import numpy as np


class CropConcat:
    def forward(self, skip, up):
        _, Cs, Hs, Ws = skip.shape
        _, Cu, Hu, Wu = up.shape
        top = (Hs - Hu) // 2
        left = (Ws - Wu) // 2
        self.skip_shape = skip.shape
        self.top, self.left, self.Hu, self.Wu = top, left, Hu, Wu
        self.skip_channels, self.up_channels = Cs, Cu
        cropped = skip[:, :, top:top + Hu, left:left + Wu]
        return np.concatenate([cropped, up], axis=1)

    def backward(self, dout):
        d_cropped = dout[:, :self.skip_channels]
        d_up = dout[:, self.skip_channels:]
        d_skip = np.zeros(self.skip_shape, dtype=dout.dtype)
        d_skip[:, :, self.top:self.top + self.Hu, self.left:self.left + self.Wu] = d_cropped
        return d_skip, d_up
