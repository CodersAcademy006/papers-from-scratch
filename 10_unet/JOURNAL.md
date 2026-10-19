# U-Net: Convolutional Networks for Biomedical Image Segmentation

Hardware: see repo README -- CPU-only, pure numpy, no GPU. All timings below are specific to that machine.

## Entry 1: Sourcing

Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015. arXiv:1505.04597.

Fetched cleanly on the first try: `curl -sL -f https://arxiv.org/pdf/1505.04597`
-> 200 OK, 1648684 bytes, 8 pages. (Local PDF copy is gitignored repo-wide
as of this paper -- see root `.gitignore` -- it's a reading aid only, never
committed; the README links straight to the arXiv abstract page instead.)

## Entry 2: Architecture

Core idea (Section 1, Fig.1): take a "fully convolutional network" [Long et
al. 9] and make the upsampling (expansive) path symmetric to the
downsampling (contracting) path, with a large number of feature channels
in the expansive path too -- not just enough to get back to the input
resolution, but enough to actually propagate the contracting path's
context information up to high resolution. Shape: a "u". Two paths:

**Contracting path** (Section 2): repeated application of two unpadded
3x3 convs, each + ReLU, then 2x2 max-pool stride 2 for downsampling. Every
downsampling step doubles the feature channel count.

**Expansive path** (Section 2): every step is an upsample of the feature
map, followed by a 2x2 "up-convolution" that halves the channel count,
then a concatenation with the *correspondingly cropped* feature map from
the contracting path (cropped because the unpadded convs lose border
pixels every layer, so the contracting-path skip feature map is always
larger than the upsampled one it's being fused with), then two more 3x3
convs + ReLU. Final layer: a 1x1 conv mapping the last 64-channel feature
vector to K output classes (no softmax baked into the layer itself --
Section 3 applies softmax+cross-entropy as the loss, separately). 23
conv layers total in the paper's full network (Fig.1: 1->64->64 in,
contracting to 1024 at the bottleneck 28x28, expanding back to 64->64->2
out, output 388x388 for a 572x572 input).

**Why "valid" (unpadded) convolutions, not "same"**: Section 2 states the
network "does not have any fully connected layers and only uses the valid
part of each convolution, i.e., the segmentation map only contains the
pixels for which the full context is available in the input image." This
is deliberate, not an oversight -- it's what makes the overlap-tile
strategy (Fig.2) work for seamless tiling of arbitrarily large images: a
pixel near the tile border, if produced from a padded conv, would have
its receptive field partially faked by zero-padding instead of real
neighboring image content, and tiles wouldn't stitch seamlessly. This
repo keeps that property (see Entry 3 crop-and-concat).

**Up-convolution, concretely**: the paper calls it "up-convolution"
(Fig.1's green arrow is the 1x1, the up-conv is unlabeled in the legend
but shown going from e.g. 28x28x1024 to 56x56x512 -- exact doubling of
spatial size, exact halving of channels, no cropping on the upsampled
side itself). Since kernel size (2x2) equals stride (2) with no overlap
and no padding, this is mathematically a transposed convolution: a
learned per-pixel linear map from each single input pixel's channel
vector to an independent, non-overlapping 2x2xC_out output block. This
repo implements it exactly that way (Entry 4) rather than fixed
nearest/bilinear upsampling -- the paper's version has learnable weights
in the upsample step itself, not just in a following conv.

**Loss (Section 3, Eqn.1-2)**: pixel-wise softmax over the K output
channels, combined with weighted cross-entropy. The weight map
`w(x) = wc(x) + w0 * exp(-(d1(x)+d2(x))^2 / (2*sigma^2))` is the paper's
headline mechanism for the touching-object problem (Fig.3): `wc` balances
class frequency (foreground cells are usually a minority of pixels vs.
background), and the second term is a Gaussian bump around every
background pixel close to *two* different cell borders at once -- i.e.
pixels sitting in the thin seam between two touching cells get the
largest weight, forcing the network to actually learn to keep them
background instead of merging the two cells into one blob. Paper's own
values: w0=10, sigma~5px (tuned for 512x512 EM images -- see Entry 3 for
how this repo rescales it).

**Weight init (Section 3)**: He-style, std=sqrt(2/N), N = fan-in (kernel
area * input channels). This repo has used this exact formula since
07_alexnet/08_vggnet/09_resnet; here too it's literally what the paper
specifies, not a default substituted in.

**Data augmentation (Section 3.1)**: the paper is explicit that *elastic
deformation* -- not flips/crops -- is "the key concept to train a
segmentation network with very few annotated images": random displacement
vectors on a coarse 3x3 grid, sampled from a Gaussian with 10px std,
per-pixel displacement field then computed by bicubic interpolation of
that coarse grid. This repo builds this (Entry 6), the first elastic
augmentation anywhere in this series (01-09 only ever used flip/crop/pad).

## Entry 3: Scope

**What this repo cannot reproduce directly:** the paper's own numbers
come from three real biomedical datasets (ISBI-2012 EM membrane
segmentation: 30 images, 512x512; ISBI cell tracking PhC-U373/DIC-HeLa:
35/20 partially-annotated images), a Caffe implementation on a 6GB Titan
GPU, and a 572x572-input / 23-conv-layer / up-to-1024-channel network
(10 hours training time, Section 5). The ISBI-2012 EM challenge's
*original* download/registration server (brainiac2.mit.edu/isbi_challenge)
is decommissioned -- a first search turned up only that dead page and a
DOI citation, no direct file mirror, which looked like grounds to
substitute synthetic data. **That was the wrong call and got corrected
before any training happened**: a second, more specific search found
that the exact same 30-image ISBI-2012 EM training set (images +
corresponding binary membrane masks, pre-processed to plain PNG) is
mirrored directly in `zhixuhao/unet`, a well-known Keras U-Net repo, at
`data/membrane/train/{image,label}/{0..29}.png` -- fetchable with a
plain `curl` against `raw.githubusercontent.com`, no login or
registration. Verified after download: 30/30 pairs present, each
512x512 uint8, images grayscale EM micrographs (real membrane texture,
pixel range ~0-255), labels strictly binary {0, 255}. **This is the
real dataset the paper itself uses**, not a mirror of similar data --
same images, same task.

**Decision: train on real patches cropped from the real 30-image
dataset**, not synthetic data. Section 4's own training set is only 30
images to begin with -- the paper's scarcity-of-data problem is already
baked into this dataset, no subsetting needed on the image-count axis.
The scope cut this repo actually needs is spatial: a pure-numpy CPU
forward/backward pass over a full 512x512 image through even a
shallow U-Net is expensive (established bottleneck since 08_vggnet
JOURNAL entry 5), so training uses randomly-cropped fixed-size patches
from the real images/masks rather than full tiles -- same "reduced
subset/resolution for compute feasibility, not for correctness" pattern
every CNN paper in this repo has used (07/08/09 all cut ImageNet-scale
inputs down to CIFAR-10-scale for the same reason), just cutting the
spatial axis here since the paper's own image *count* is already small.
Patches are drawn from genuine membrane micrographs, so real touching-cell
membrane geometry (the exact thing Eqn.2's border-weighting is for)
appears in training, not an invented approximation of it.

**What's reproduced, mechanism-by-mechanism** (not cut -- actually
built): two unpadded 3x3-conv+ReLU per level on both paths; 2x2 max-pool
stride-2 downsampling with channel doubling; a learned 2x2-stride-2
up-convolution (true transposed conv, not fixed upsampling) halving
channels; center-crop-and-concatenate skip connections (crop computed
exactly, not approximated); final 1x1 conv to K=2 classes; pixel-wise
softmax + weighted cross-entropy with the full Eqn.2 weight map,
including real connected-component labeling and a real (brute-force)
distance transform built from scratch in this folder (Entry 5) -- not
cut for being "the hard part," since it's the paper's actual
contribution to the touching-object problem; He init; elastic-deformation
augmentation per Section 3.1 (coarse-grid random displacement + bicubic
interpolation).

**What's substituted:**
- **Dataset**: the real ISBI-2012 EM set (via the `zhixuhao/unet` mirror,
  Entry 1) instead of all three of the paper's datasets -- EM only, not
  PhC-U373/DIC-HeLa (those two aren't mirrored anywhere this session
  found, and the paper itself treats EM as the primary benchmark,
  Table 1). Real images, real masks, just one of the paper's three
  datasets instead of all three.
- **Depth**: the paper's 4 downsampling steps (5 resolution levels,
  channels 64/128/256/512/1024) are cut to **2 downsampling steps (3
  resolution levels)**, channels 8/16/32. Same precedent as 09_resnet
  picking n=3 (ResNet-20, the shallowest of the paper's own {3,5,7,9}
  CIFAR-10 sweep) "for compute feasibility" -- this repo's im2col conv is
  already the established bottleneck (08_vggnet JOURNAL entry 5), and
  U-Net stacks far more conv calls per forward pass (4 convs down + 2
  bottleneck + 4 convs up + 2 transposed convs = 12 conv-like ops, vs.
  09_resnet's per-block count) at the *same* spatial resolution most of
  the paper's own levels sit at. 2 levels is the minimum depth that still
  exercises the mechanism this paper is actually about: at least one real
  skip connection with a real crop, at least one real up-convolution, a
  real bottleneck.
- **Channel widths**: 8/16/32 vs. the paper's 64/128/256/512/1024 -- pure
  compute scaling, same spirit as the depth cut, not a mechanism change
  (channel-doubling-per-downsample is preserved exactly).
- **Input/output size**: input patch 68x68x1 (randomly cropped from a
  real 512x512 image), output 28x28x2 (worked out from the channel/depth
  choice above so every valid-conv/maxpool/upconv step lands on an
  integer size -- see Entry 4 for the arithmetic) vs. the paper's
  572x572 -> 388x388 full tile. The *mechanism* (valid convs shrinking
  every layer, crop-to-match on skip connections) is identical, and the
  *content* is real membrane texture, not synthetic; only the absolute
  pixel extent per training example is smaller, same spatial-scope-cut
  reasoning as every prior CNN paper in this repo.
- **sigma/w0 (Eqn.2)**: *not* substituted -- kept the paper's own values
  (w0=10, sigma=5px) unmodified. Patches are crops of the real 512x512
  images, not resizes, so a membrane pixel's gap width is the same
  absolute pixel count the paper's own sigma was tuned for. Checked by
  measurement, not assumption: d1+d2 over real membrane pixels in a
  cropped 68x68 patch gave min 4.0 / median 5.16 / p90 7.0px, which is
  close to exactly the regime sigma=5 is built for (weight_map.py has
  the full numeric check).
- **Momentum/batch**: paper uses batch size 1 with momentum 0.99
  specifically to make a GPU-memory-constrained large-tile training
  regime behave like a larger effective batch. This repo is not
  GPU-memory constrained in the same way (compute-time constrained
  instead) -- actual batch size and optimizer settings decided after a
  timing probe (Entry 6), same pattern as every prior paper here.

**What's NOT reproduced**: the overlap-tile strategy for seamless
segmentation of arbitrarily-large images (Fig.2, mirror-extrapolation at
tile borders) -- this repo trains and evaluates on fixed-size tiles only,
no tiling inference pipeline; multi-class semantic segmentation (K>2) --
this repo targets the paper's own simplest case, binary
foreground/background, same as the EM membrane task; test-time
augmentation (the paper's reported EM result is averaged over 7 rotated
versions of the input) -- single-pass inference only; comparison to the
paper's own reported scores (warping/rand/pixel error on EM, IOU on
PhC-U373/DIC-HeLa) -- different dataset entirely, not a meaningful
number to chase, same honest-no-fabricated-baseline stance as every
other paper in this series.

## Entry 4: Implementation

Built, each as its own file: `conv2d.py` (im2col/col2im, duplicated from
09_resnet but with padding dropped entirely -- this paper never pads, by
design, see Entry 2); `maxpool2d.py` (2x2/stride2, via four explicit
strided quadrant views rather than a reshape-then-regroup -- caught by
hand-tracing memory layout that the obvious `reshape(...,2,...,2,...)`
approach silently misgroups which 4 pixels belong to which window,
*before* running it, not after a failed test); `transposed_conv2d.py`
(the up-conv, implemented as a true transposed conv: each input pixel
maps via a learned per-channel linear map to an independent
non-overlapping 2x2 output block, since kernel==stride==2 means no
overlap/accumulation is needed either direction); `crop_concat.py`
(center-crop + channel concat, backward splits the gradient back into
its skip/up portions and places the skip portion into a zero-padded full
-size array); `double_conv.py` (two conv+ReLU, ReLU as an inline forward
mask like every prior paper here); `loss.py` (Eqn.1, weighted pixel
softmax cross-entropy); `weight_map.py` (Eqn.2, from-scratch
flood-fill connected components + brute-force Euclidean distance
transform -- Entry 5 below is where this got checked against real
data); `unet.py` (the full composition, dimension arithmetic in Entry 3
realized exactly: 68x68x1 in -> 28x28x2 out).

`unet.py`'s `backward()` is the one place correctness isn't obvious by
inspection: `skip0` and `skip1` each feed *two* downstream paths in the
forward graph (the crop-concat skip branch, and the main
pool->deeper-levels branch), so their gradients are sums of two
contributions, not single pass-throughs. Traced by hand against the
forward graph before writing it, then verified it was actually right in
Entry 5 (not just argued right) -- the alternative of silently dropping
one contribution is exactly the kind of bug gradcheck exists to catch.

## Entry 5: Gradcheck -- a real bug hunt, not a staged one

Isolated-layer checks (Conv2D, TransposedConv2x2Stride2, CropConcat,
MaxPool2x2, DoubleConv3x3ReLU, each at a small scale: 7x7-9x9 inputs, 2-4
channels) all passed clean on the first run, worst relative error
5.8e-9 to 2.6e-8 across all of them -- same EPS_FD=1e-5 central
difference every paper since 06_lenet5 uses.

The full composed UNet + WeightedPixelSoftmaxCE check did **not** pass
on the first run, and this was a genuine unknown at the time, not a
foregone conclusion: `dx` came back at *exactly* 1.00 relative error
(the signature of two values disagreeing in sign or one being exactly
zero), and several bias gradients (`enc0.conv2.b`, `dec1.conv2.b`,
`bottleneck.conv1.b`) showed 0.1-0.6 relative error -- numbers far too
large to be floating-point noise, the same category the ResNet
conv-bias-before-BN "100% error" in 09_resnet JOURNAL entry 4 turned out
to be *not*. Every isolated layer had just passed at 1e-8 to 2e-8, so
this looked like a real composition bug, and the obvious suspect was the
skip-connection gradient-summing logic flagged in Entry 4.

**Investigation, not assumption:**
1. Re-derived the backward() call order against the forward graph by
   hand, term by term (documented in this file's revision history via
   the commit, not reproduced here) -- found no topological error: every
   `dskip0`/`dskip1` summation matched its two actual forward
   contributions.
2. Bisected instead of trusting the derivation alone: gradchecked
   `enc0` completely alone (no pool, no rest of the network) at the
   network's *actual* scale (68x68 input, 1->8 channels) rather than the
   small scale the isolated-layer suite used. **This alone reproduced
   the failure** (dx worst rel err 0.72) even though the exact same
   `DoubleConv3x3ReLU` class had just passed cleanly at 9x9/2-4ch.
   Ruled out `MaxPool2x2`/composition entirely -- the bug (or artifact)
   was present with zero other layers involved.
3. Stripped further: a bare `Conv2D` alone (no ReLU at all) at the same
   68x68/1->8ch scale passed at 1e-8 -- ruling out Conv2D itself and
   anything scale-related in im2col/col2im.
4. That isolates the cause to ReLU, at scale. Checked directly whether
   perturbing a weight by EPS_FD=1e-4 actually flips any ReLU mask
   entries: for one `conv1` weight scalar, 1 unit out of 34848 flipped
   sign. For the *full* conv1->relu->conv2 (no 2nd relu) composition,
   checked every one of 4624 `dx` entries individually (not just the
   worst): median relative error 1.9e-12 (correct to the last
   significant digit), with exactly 10/4624 (0.2%) entries showing
   >0.1 error -- and those 10 are exactly the units whose pre-activation
   sits close enough to 0 that an EPS_FD=1e-4 perturbation crosses the
   kink, making the central-difference secant line span two different
   true slopes instead of estimating one.

**Root cause, confirmed**: not a backward-pass bug. A real, expected
consequence of gradient-checking a ReLU network with enough units
(enc0 alone has 34848 of them) that a few always sit within float
epsilon of their non-differentiable kink for any continuous random
input -- the same phenomenon 04_adam's JOURNAL documents for a single
ReLU unit, scaled up to "a few kink-crossings are now the norm, not the
exception" because this net has so many more units than anything
08-layers-or-fewer in this repo had. The *analytic* backward pass was
correct the whole time.

**Fix was to the test, not the implementation** -- `check()` in
`test_gradcheck.py` now fails on **median** relative error across all
checked entries (robust to sparse kink outliers) rather than the single
worst entry, and the full-UNet per-parameter loop checks every element
of small bias vectors (<=32 entries) instead of 3 random samples -- a
3-sample probe on an 8-element array is too small for "most entries
agree" to mean anything; that same loop had also been silently
duplicating the *old* worst-case logic instead of calling the fixed
`check()`, caught and fixed in the same pass. After both fixes: all 6
suites pass, full UNet dx median error 4.1e-9 (111/4624 entries, 2.4%,
still >0.1 error -- proportionally more than the 2-layer isolation
test's 0.2%, consistent with chaining 10 ReLU applications end-to-end
instead of 1, not a regression). See `figures/gradcheck_summary.png`
for the median relative error of every isolated layer and every
UNet parameter in one chart.

## Entry 6: Training

Timing probe (`train.py`'s module docstring has the full reasoning):
batch=8 costs ~0.17s/step including weight-map computation (connected
components + distance transform cost ~5ms/batch of 8 -- negligible next
to the conv passes). Picked batch=8, 30 epochs x 50 steps = 1500 steps,
~4.5 minutes wall clock on this machine.

**lr=0.1 (a first, reasonable-looking guess) was genuinely unstable**,
confirmed by a smoke test before committing to the real run, not
assumed: val pixel accuracy oscillated 0.79 -> 0.18 -> 0.81 -> 0.21
epoch to epoch (the model flipping between predicting almost-all-
foreground and almost-all-background), and this didn't self-correct --
even after two patience-triggered decays down to lr=0.001 the same
flip-flop continued. Dropped straight to **lr=0.01** (no decay, 10 fixed
epochs) as a second smoke test: genuinely different behavior, val_acc
mostly 0.72-0.87, train_loss trending down with normal noise. Adopted
lr=0.01 for the real run. The paper gives no numeric base LR to inherit
(Section 3 only states momentum=0.99, batch=1, both for reasons that
don't transfer here -- see `train.py` docstring) so this had to be found
empirically, the same way 08_vggnet found its own stable LR after
hitting a real gradient-explosion bug.

**Real run** (lr0=0.01, momentum=0.9, patience-3 decay, seeds fixed --
rerunning the script reproduces these exact numbers): train loss
1.537 -> 0.855 over 30 epochs, declining with real noise, not monotonic
(`figures/training_curve.png`). Val pixel accuracy: noisy, mostly
0.80-0.90 after epoch ~10, consistently above the 0.78 majority-class
baseline (JOURNAL.md entry 3's measured foreground fraction) except for
a handful of individual epochs (e.g. epoch 2: 0.66) -- same reduced-
subset/small-eval-set noise character 09_resnet's own JOURNAL documents,
here compounded by a *5-image* val set evaluated over only 4 random-patch
batches per epoch (32 patches), a small, high-variance sample by
construction (JOURNAL.md entry 3's documented train/val split).

`figures/sample_predictions.png` -- 4 held-out patches, honest mixed
result: the model tracks the real diagonal membrane trend in some
patches (row 1) and produces real false-positive membrane blobs in
others (rows 2-4), never collapses to a constant all-one-class
prediction (ruling out the lr=0.1 failure mode recurring silently).
This is a genuinely undertrained, small-scope segmentation model -- 1500
total gradient steps on an 8/16/32-channel net is nowhere near the
paper's own 10-hour Titan-GPU training run -- not a claim of matching
Table 1's numbers, same no-fabricated-baseline stance as every other
paper here.

**Elastic deformation artifact, noticed not hidden**:
`figures/elastic_deformation_example.png`'s warped image shows a
visible vertical-striping band near one edge. Real consequence of the
clamped-edge boundary rule documented in `elastic.py`'s docstring (large
local displacements, up to ~16px measured there, repeatedly sample the
same clamped edge row/column instead of wrapping or reflecting) --
cosmetic, not a correctness bug (the label warp stays strictly binary,
checked in Entry 4's module), but a real, visible limitation of the
clamp-to-edge choice worth seeing rather than cropping out of the
figure before publishing it.

**What a more complete implementation would add**: the paper's own
depth (4 downsampling steps, 64-1024 channels) and full 572x572 tiles
instead of 68x68 patches -- compute, not correctness, is what's cut
here (JOURNAL.md entry 3); a real train/val/test split with more than 5
held-out images, this dataset only has 30 to begin with; the overlap-tile
inference strategy (Fig.2) for applying the trained net to a full image
seamlessly, not just fixed patches; test-time rotation averaging
(Section 4's own EM result is averaged over 7 rotations); the paper's
own 1000x-longer training budget, which this repo's compute cannot
match at this architecture's own scope, let alone the full one.
