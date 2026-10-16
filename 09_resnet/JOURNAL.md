# ResNet (Deep Residual Learning for Image Recognition)

Hardware: see repo README -- CPU-only, pure numpy, no GPU. All timings below are specific to that machine.

## Entry 1: Sourcing

He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition",
CVPR 2016. arXiv:1512.03385.

Fetched cleanly on the first try: `curl -sL -f https://arxiv.org/pdf/1512.03385`
-> 200 OK, 819383 bytes, 12 pages.

## Entry 2: Architecture

Core idea (Section 3.1, Eqn.1, Fig.2): instead of a stack of layers
learning a desired mapping H(x) directly, let them learn a residual
F(x) := H(x) - x, so the block computes y = F(x,{Wi}) + x ("shortcut
connection" + element-wise addition, identity, no extra parameter or
compute). Motivation (Section 1): plain (non-residual) networks exhibit
a *degradation* problem when made deeper -- not overfitting, but higher
*training* error, which a solver should be able to avoid by construction
(stack identity mappings on top of a shallower solution) but empirically
doesn't find on its own. Residual reformulation makes identity the
solver's default/zero-perturbation case instead of something it has to
discover from scratch.

Section 3.3 describes the paper's ImageNet-scale networks (Table 1:
18/34/50/101/152-layer variants, the 50+ layer ones using a 3-layer
"bottleneck" block, Fig.5 right). This repo does not target that table --
see entry 3.

Section 3.2 / Eqn.2: when shortcut input/output dimensions differ (channel
count changes), either (A) zero-pad the identity mapping (no extra
parameter) or (B)/(C) a 1x1 conv projection W_s. Section 4.2 (CIFAR-10)
states plainly: "we use identity shortcuts in all cases (i.e., option A)"
-- this repo follows that, see entry 3.

**Section 4.2 (CIFAR-10 and Analysis) is what this repo actually builds**
-- unlike AlexNet/VGG, this paper describes its own from-scratch CIFAR-10
architecture, not an ImageNet-only design this repo has to rescale down:

- Input 32x32, per-pixel mean subtracted.
- First layer: 3x3 conv, 16 filters.
- Then a stack of 6n layers with 3x3 convs on feature maps of sizes
  {32,16,8} respectively, 2n layers for each map size, filter counts
  {16,32,64} respectively. Subsampling (halving spatial size) is done by
  the first conv of a stage, stride 2.
- Shortcuts connect pairs of 3x3 layers (3n shortcuts total). Identity
  shortcuts in all cases (option A): "the shortcut still performs identity
  mapping, with extra zero entries padded for increasing dimensions...
  performed with a stride of 2" when crossing a size change.
- Ends with global average pooling, a 10-way FC, softmax. Total weighted
  layers: 6n+2.
- Paper compares n={3,5,7,9} -> 20/32/44/56-layer nets (Table 6), plus a
  separate 110-layer (n=18, needs an LR warm-up trick) and a 1202-layer
  (n=200) exploratory run.

Training recipe (Section 4.2): weight decay 0.0001, momentum 0.9, weight
init as in He et al. 2015 [13] (He init -- this repo already uses this
for 07_alexnet/08_vggnet, but here it's not a substitution, it's literally
what the paper specifies), BN after every conv and before activation, no
dropout. Mini-batch 128 on two GPUs. LR starts 0.1, divided by 10 at 32k
and 48k iterations, training terminated at 64k iterations, on a 45k/5k
train/val split. Data augmentation (following [24]): pad 4 pixels each
side, randomly sample a 32x32 crop from the padded image or its
horizontal flip. Testing: single view of the original 32x32 image (no
multi-crop).

## Entry 3: Scope

The CIFAR-10 recipe above is native to the paper -- the first paper in
this CNN section where no ImageNet-to-CIFAR rescale has to be invented
(07_alexnet and 08_vggnet both had to make real architectural judgment
calls to fit an ImageNet-only design onto 32x32; here Section 4.2 already
targets 32x32 directly).

**Depth choice:** this repo builds **n=3 -> ResNet-20** (6*3+2=20 weighted
layers), the shallowest of the paper's own CIFAR-10 comparison set. Picked
for compute feasibility: 07_alexnet/08_vggnet's JOURNAL entries already
established that this machine's pure-numpy im2col conv is the compute
bottleneck (08_vggnet entry 5: ~30s/batch-128 on an 11-weight-layer net),
and BN forward+backward at every conv is overhead this repo hasn't paid
before (03_batchnorm was an MLP, not a conv net). More layers directly
means more wall time; 20 is the cheapest point on the paper's own n
sweep that still demonstrates the full mechanism (basic blocks, option-A
shortcuts, 3 feature-map stages).

**What's reproduced, mechanism-by-mechanism** (not cut -- actually built):
conv1 (3x3,16) -> 3 stages of 2n=6 BasicBlocks each, filter widths
{16,32,64}, feature maps {32,16,8}, first block of stage 2/3 strided 2 for
downsampling; option-A identity shortcuts exclusively (stride-2 spatial
subsample + zero-pad channels, parameter-free) -- this isn't a
simplification, it's the paper's own chosen variant for CIFAR-10, not
option B/C; BN after every conv, before ReLU, He init; SGD + momentum 0.9
+ weight decay 1e-4 (the CIFAR-10 recipe's own value, distinct from
07_alexnet/08_vggnet's ImageNet-recipe 5e-4); global average pool + FC-10
+ softmax; pad-4 / random-32x32-crop / random-hflip augmentation --
**the first paper in this series where augmentation is actually
implemented**, rather than scope-cut, since Section 4.2 specifies it
directly for this exact 32x32 dataset and it's cheap (a data-pipeline op,
not an added compute cost on the conv layers).

**What's substituted:** the paper's iteration-count LR schedule (divide by
10 at 32k/48k iterations, stop at 64k) is tuned for a 45k-image training
set over many epochs -- doesn't map onto a reduced-subset/few-epoch run.
Same patience-based decay precedent 07_alexnet/08_vggnet already use.
Mini-batch 128 on two GPUs -> single CPU process, actual batch size
decided after a timing probe (next entry). Zero-padding is used for the
4-pixel border (the paper doesn't specify the pad mode); reflect-padding
would also be defensible but zero is the simpler default and what [24]
(Lee et al., deeply-supervised nets, the augmentation's cited source)
uses.

**What's NOT reproduced:** n={5,7,9,18,200} (32/44/56/110/1202-layer
variants, Table 6) and the "exploring over 1000 layers" study -- this
repo targets one depth, same single-configuration spirit as 08_vggnet
being config-A-only; option B/C shortcuts -- not even a cut, the paper
itself prefers option A for CIFAR-10; the 45k/5k train/val split used for
the paper's own hyperparameter exploration -- this repo's training subset
follows the same reduced-subset precedent 07_alexnet/08_vggnet already
established, full 10k test set still used for honest evaluation; Section
4.3 (object detection on PASCAL/COCO via Faster R-CNN) and the ImageNet
Table 1-5 results -- different systems/datasets, out of scope as always.

## Entry 4: Gradcheck

`test_gradcheck.py` (isolated layers, central-difference, EPS_FD=1e-5):
Conv2D at stride=2 (the downsampling convs 08_vggnet never exercised,
since VGG is stride-1-only), BatchNorm2d (the spatial wrapper over
BatchNorm1d), IdentityShortcut (option A: subsample + zero-pad), and a
full BasicBlock crossing a stage boundary (stride=2, channel count
doubling, so the shortcut path is actually exercised). All passed clean,
worst relative error 3.0e-8 across every one.

`test_resnet_gradcheck.py` (full composed ResNet20_CIFAR, EPS_FD=1e-8,
same tightened epsilon 08_vggnet's full-net check used): every BN
gamma/beta and every conv weight passed at 1e-5 to 1e-7 relative error.
But every conv **bias** immediately followed by a BatchNorm layer --
conv1.b, and every block's conv1.b/conv2.b -- showed "100% relative
error" (analytic and numeric both ~1e-15 to 1e-16, i.e. both zero up to
floating-point noise, so the relative-error ratio is meaningless there;
the assert's absolute-tolerance branch is what actually passes them).

This isn't a bug, it's an exact mathematical property of this specific
architecture: BatchNorm2d normalizes each channel to zero mean over the
batch+spatial axes, and that normalization is invariant to any additive
shift applied uniformly across the whole batch for that channel -- which
is exactly what perturbing a conv bias does. Working it out from
bn_layer.py's own backward() (summing dx over the batch axis): the
dmu_B and dsigma_B_sq terms are built so that sum_i(dx_i) collapses to
std_inv*sum(dx_hat_i) - std_inv*sum(dx_hat_i) = 0 identically, for any
conv output that feeds straight into a BN layer. So d(loss)/d(bias) = 0
exactly, in both the analytic backward pass and the true (noise-free)
numerical derivative -- the "100%" is two independent near-zero
estimates disagreeing on sign/magnitude in the 15th decimal place, not a
wrong gradient. Confirmed by printing the actual analytic magnitudes:
~1.3e-15 for conv1.b, ~6.9e-16 for a block's conv1.b -- machine-epsilon
noise, not a real gradient.

Separately: the first run of the full-net check threw a `RuntimeWarning:
divide by zero / overflow / invalid value encountered in matmul` from
dense.py's `dW = da.T @ self.x` line, with -W error showing it fires on
the very first (unperturbed) backward call, before the gradcheck loop
even starts. Traced it by hand: printed every intermediate activation
through the whole forward pass (conv1 through all 9 blocks to the pooled
features and logits) -- no NaN/Inf anywhere, and the resulting dW itself
contains no NaN/Inf either, just a spurious warning. Reproduced in
isolation with unrelated random arrays of the same shapes: any `(2,10)
.T @ (2,64)`-shaped transposed matmul on this machine fires the same
warning, even outside this codebase entirely. This is numpy's Apple
Accelerate BLAS backend (`numpy.show_config()` confirms `accelerate` as
the blas/lapack provider) raising a hardware FP-exception flag on a
non-contiguous-transpose matmul path that doesn't actually affect the
output -- a known Accelerate/numpy false positive on Apple Silicon, not
a defect in this repo's math. Confirmed harmless and left as-is (the
gradcheck script doesn't filter it, since filtering a warning one hasn't
verified is benign would be the wrong default -- this one specifically
has been).

## Entry 5: Final training results

Config: 8 epochs, 5000-example training subset (same reduced-subset
precedent as 08_vggnet), batch 128, full 10000-example test set for
evaluation. SGD momentum 0.9, weight decay 1e-4, lr=0.1 -- the paper's
own CIFAR-10 starting LR (Section 4.2), unlike 08_vggnet which had to cut
its borrowed ImageNet-recipe LR by 100x after an actual gradient-explosion
bug. Here lr=0.1 is stable from the first batch: a timing probe on random
data showed sane ~3.1 loss with no blowup, and a full real smoke-test
epoch confirmed it (loss 2.78->1.57 across the epoch, no NaN) before the
real run was launched. First real sign this paper's own hyperparameters
are actually scaled for CIFAR-10 already, not borrowed from ImageNet.

Train NLL: smooth, monotonic every epoch -- 2.3444, 1.8494, 1.7144,
1.6707, 1.5793, 1.5052, 1.4669, 1.4612.

Test error: 74.44%, 69.57%, 64.85%, 69.29%, 60.93%, 55.42%, 70.88%,
56.13%. Overall declining (74.44% -> 56.13%) but genuinely noisy
epoch-to-epoch, not monotonic like 07_alexnet/08_vggnet's test curves
were -- best was epoch 5 (55.42%), epoch 6 jumped back up to 70.88%
before epoch 7 recovered to 56.13%. lr never decayed: the patience-3
counter reset on every new best and only ever reached 2 consecutive
non-improving epochs (epoch 3, then epoch 6/7), never the 3 needed to
trigger a decay -- the log confirms lr=0.10000 unchanged across all 8
rows.

Honest read of the noise: this run never gets the stabilizing LR decay
the paper's own schedule relies on (divide by 10 at 32k/48k of 64k total
iterations -- this run's entire 8-epoch budget is roughly 312 iterations
at batch 128 over a 5000-example subset, nowhere near that schedule's
regime to begin with, the same subset-vs-iteration-count scope cut
08_vggnet/07_alexnet already made explicit in entry 3). Three compounding
sources of batch-to-batch variance at this reduced scale: (1) BatchNorm's
batch statistics are computed from a 128-example minibatch, noisier than
the paper's own batches on a 45k-image set; (2) the pad-4/crop/hflip
augmentation randomizes every single batch, so no two epochs see the same
inputs even over the same 5000-image subset; (3) a constant high LR
(0.1, never decayed here) keeps the optimizer taking large steps for the
entire run instead of settling as training progresses. None of this is a
bug -- gradcheck (entry 4) already confirmed the backward pass is exact;
it is what training a 20-layer net at the paper's own LR, with no
warmup and no decay, on a 10%-sized subset for 8 epochs actually looks
like once run for real.

Not claiming comparison to the paper's own reported number (~8.75% test
error for ResNet-20, full 45k/5k split, 64k iterations with LR decay --
Table 6): different training budget by roughly two orders of magnitude,
same honest-no-fabricated-baseline stance as every other paper in this
series.
