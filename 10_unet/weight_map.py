"""
Section 3, Eqn.2 -- the paper's actual headline mechanism for the
touching-object problem:

    w(x) = wc(x) + w0 * exp( -(d1(x) + d2(x))^2 / (2*sigma^2) )

wc: per-pixel class-balance weight (compensates foreground/background
pixel-frequency imbalance). d1(x): distance from pixel x to the border of
the nearest cell. d2(x): distance to the border of the second-nearest
cell. The Gaussian term is large exactly where a background pixel sits
close to *two* different cells' borders at once -- the thin separating
seam between touching cells -- which is what forces the network to keep
predicting background there instead of merging the two cells.

Built from scratch, same "no shortcut library for the mechanism itself"
rule as the rest of this repo: connected-component labeling via
breadth-first flood fill (not scipy.ndimage.label), and the distance
field via brute-force nearest-border-pixel search (not
scipy.ndimage.distance_transform_edt). Both are fine at this repo's
scoped image sizes (JOURNAL.md entry 3): O(H*W * total_border_pixels)
for the distance search, trivial at ~68x68.

sigma: paper uses sigma~5px, tuned on the real 512x512 EM images. This
repo trains on 68x68 patches *cropped* (not resized) from those same
real images (JOURNAL.md entry 3), so a membrane pixel's gap width in a
patch is the exact same absolute pixel count it would be in the full
image -- there is no resolution change to rescale sigma for, unlike the
channel/depth cuts elsewhere in this architecture. Verified directly
rather than assumed: measuring d1+d2 over real membrane pixels in a
cropped 68x68 patch (label 0, patch [200:268,200:268]) gives min 4.0,
median 5.16, p90 7.0 pixels -- almost exactly the regime sigma=5 is
built for (exp(-(d1+d2)^2/(2*sigma^2)) at these three values: 0.73,
0.59, 0.38 with sigma=5, vs. 0.14, 0.04, 0.002 with a naively
downscaled sigma=2 -- the smaller sigma would have made the Gaussian
term nearly dead across the real gap-width distribution this dataset
actually has). **W0 and SIGMA below are the paper's own unmodified
values** -- the one Eqn.2 constant this repo didn't have to touch.
"""
import numpy as np
from collections import deque

W0 = 10.0
SIGMA = 5.0


def label_components(mask):
    """4-connected flood fill. mask: (H,W) bool, foreground=True. Returns
    (H,W) int array, 0=background, 1..M=component ids."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 0
    for i0 in range(H):
        for j0 in range(W):
            if mask[i0, j0] and labels[i0, j0] == 0:
                next_label += 1
                q = deque([(i0, j0)])
                labels[i0, j0] = next_label
                while q:
                    i, j = q.popleft()
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] and labels[ni, nj] == 0:
                            labels[ni, nj] = next_label
                            q.append((ni, nj))
    return labels, next_label


def component_border_pixels(labels, comp_id):
    """Pixels belonging to comp_id that touch background or the image edge."""
    H, W = labels.shape
    is_comp = labels == comp_id
    border = np.zeros_like(is_comp)
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.zeros_like(is_comp)
        si0, si1 = max(0, -di), H - max(0, di)
        sj0, sj1 = max(0, -dj), W - max(0, dj)
        di0, di1 = max(0, di), H - max(0, -di)
        dj0, dj1 = max(0, dj), W - max(0, -dj)
        shifted[di0:di1, dj0:dj1] = is_comp[si0:si1, sj0:sj1]
        border |= is_comp & ~shifted
    # pixels on the image edge are always a border (no neighbor beyond edge)
    border[0, :] |= is_comp[0, :]
    border[-1, :] |= is_comp[-1, :]
    border[:, 0] |= is_comp[:, 0]
    border[:, -1] |= is_comp[:, -1]
    return np.argwhere(border)  # (B, 2) array of (row, col)


def distance_to_borders(mask, labels, n_components):
    """d1, d2: (H,W) float -- Euclidean distance from every pixel to the
    nearest and second-nearest component's border. If fewer than 2
    components exist, d1/d2 are set to a large sentinel (no touching-pair
    effect possible)."""
    H, W = mask.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)  # (H*W, 2)

    if n_components < 2:
        big = np.full((H, W), 1e6)
        return big, big.copy()

    per_component_dist = np.empty((n_components, H * W), dtype=np.float64)
    for c in range(1, n_components + 1):
        border = component_border_pixels(labels, c).astype(np.float64)  # (B,2)
        diff = coords[:, None, :] - border[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        per_component_dist[c - 1] = dist

    sorted_dist = np.sort(per_component_dist, axis=0)
    d1 = sorted_dist[0].reshape(H, W)
    d2 = sorted_dist[1].reshape(H, W)
    return d1, d2


def class_balance_weight(mask):
    """wc(x): inverse-frequency weight per class, normalized so the
    average weight over the image is 1.0."""
    H, W = mask.shape
    n_fg = mask.sum()
    n_bg = H * W - n_fg
    wc = np.empty((H, W), dtype=np.float64)
    if n_fg == 0 or n_bg == 0:
        return np.ones((H, W), dtype=np.float64)
    freq_fg, freq_bg = n_fg / (H * W), n_bg / (H * W)
    w_fg, w_bg = 1.0 / (2 * freq_fg), 1.0 / (2 * freq_bg)
    wc[mask] = w_fg
    wc[~mask] = w_bg
    return wc


def compute_weight_map(mask):
    """mask: (H,W) bool, foreground=True. Returns w(x): (H,W) float,
    Eqn.2 in full."""
    labels, n_components = label_components(mask)
    d1, d2 = distance_to_borders(mask, labels, n_components)
    wc = class_balance_weight(mask)
    gaussian = W0 * np.exp(-((d1 + d2) ** 2) / (2 * SIGMA ** 2))
    return wc + gaussian
