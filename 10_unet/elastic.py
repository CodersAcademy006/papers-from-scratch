"""
Elastic deformation augmentation, Section 3.1: "random displacement
vectors on a coarse 3 by 3 grid. The displacements are sampled from a
Gaussian distribution with 10 pixels standard deviation. Per-pixel
displacements are then computed using bicubic interpolation." The paper
calls this "the key concept to train a segmentation network with very
few annotated images" -- not optional polish, so built as its own real
mechanism rather than substituted with a flip/crop (which is all this
repo has used for augmentation through 09_resnet).

Built in two from-scratch pieces:
1. Cubic-convolution upsampling (the standard separable algorithm real
   bicubic image resizing uses: a=-0.5 cubic kernel, applied along rows
   then columns) to go from a sparse 3x3 random grid to a smooth dense
   per-pixel displacement field. Edge control points are clamped rather
   than mirrored/wrapped -- simplest defensible boundary rule for a 3x3
   grid, where a 4-tap cubic stencil runs out of real neighbors almost
   everywhere anyway.
2. The actual warp: output pixel (y,x) samples the input at
   (y+dy(y,x), x+dx(y,x)). Image pixels are bilinearly sampled (standard
   choice for continuous-valued image warps). Label pixels are
   nearest-neighbor sampled instead -- bilinear would blend the binary
   {0,1} class values at every border, inventing fractional "half-cell"
   labels that don't exist in Eqn.1's cross-entropy formulation.

grid_std: paper's 10px is calibrated for 512x512 full images. This repo
deforms the already-cropped 68x68 patch directly (simpler than deforming
the full 512 image then re-cropping, and the displacement field is local
to the patch either way), so grid_std is scaled down from the paper's
10px -- not by the naive 10*(68/512)~1.3px linear ratio, which produces
a barely-visible warp at this patch size (tried first, rejected by eye:
nearly identical to the unwarped patch). Settled on grid_std=4.0 by
checking the actual resulting displacement field: run 200 random (3x3,
std=4) fields through `resize_grid_bicubic` and measured
sqrt(dy^2+dx^2).max() per field -- worst case over the 200 was 16.2px,
i.e. genuinely large local warps do happen, not just gentle ones. That's
fine, not a bug to guard against: `_bilinear_sample`/`_nearest_sample`
clamp out-of-bounds source coordinates to the nearest edge pixel rather
than indexing out of range, so a large displacement just samples (and
duplicates) edge content instead of crashing or wrapping around.
"""
import numpy as np

GRID_STD = 4.0


def _cubic_kernel(x, a=-0.5):
    x = np.abs(x)
    x2, x3 = x * x, x * x * x
    w = np.where(
        x <= 1, (a + 2) * x3 - (a + 3) * x2 + 1,
        np.where(x < 2, a * x3 - 5 * a * x2 + 8 * a * x - 4 * a, 0.0),
    )
    return w


def _resize_1d_weights(n_in, n_out):
    """Returns (n_out, 4) weights and (n_out, 4) clamped source indices
    for cubic-convolution upsampling from n_in samples to n_out samples."""
    src_pos = np.arange(n_out) * (n_in - 1) / (n_out - 1)
    i0 = np.floor(src_pos).astype(int)
    frac = src_pos - i0
    offsets = np.array([-1, 0, 1, 2])
    idx = np.clip(i0[:, None] + offsets[None, :], 0, n_in - 1)
    w = _cubic_kernel(frac[:, None] - offsets[None, :])
    w = w / w.sum(axis=1, keepdims=True)
    return w, idx


def resize_grid_bicubic(grid, out_h, out_w):
    """grid: (n,n) coarse control points -> (out_h,out_w) dense field."""
    n = grid.shape[0]
    wr, ir = _resize_1d_weights(n, out_h)
    tmp = (grid[ir, :] * wr[:, :, None]).sum(axis=1)  # (out_h, n)
    wc, ic = _resize_1d_weights(n, out_w)
    out = (tmp[:, ic] * wc[None, :, :]).sum(axis=2)  # (out_h, out_w)
    return out


def _bilinear_sample(img, ys, xs):
    H, W = img.shape
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wy = np.clip(ys - y0, 0, 1)
    wx = np.clip(xs - x0, 0, 1)
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def _nearest_sample(img, ys, xs):
    H, W = img.shape
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    return img[yi, xi]


def elastic_deform(image, label, grid_size=3, grid_std=GRID_STD, rng=None):
    """image: (H,W) float. label: (H,W) int, same shape. Returns warped
    (image, label), same shapes, using one shared displacement field."""
    rng = rng or np.random.default_rng()
    H, W = image.shape
    dy_grid = rng.normal(0, grid_std, size=(grid_size, grid_size))
    dx_grid = rng.normal(0, grid_std, size=(grid_size, grid_size))
    dy = resize_grid_bicubic(dy_grid, H, W)
    dx = resize_grid_bicubic(dx_grid, H, W)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    src_y = yy + dy
    src_x = xx + dx

    warped_image = _bilinear_sample(image, src_y, src_x)
    warped_label = _nearest_sample(label, src_y, src_x)
    return warped_image, warped_label
