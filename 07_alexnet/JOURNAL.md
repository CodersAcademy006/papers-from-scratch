# 07_alexnet — Journal

Hardware: see repo README — CPU-only, pure numpy, no GPU. All timings below are specific to that machine.

## Entry 1 — Sourcing

Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep
Convolutional Neural Networks," NeurIPS 2012. Not on arXiv (pre-dates
arXiv's ML norm; published via NeurIPS proceedings). Fetched directly
from papers.nips.cc's own PDF host (9 pages) — worked on the first try,
no retry needed this time (contrast with 06_lenet5's author-site timeout).

## Entry 2 — The architecture, by quote

Section 3.5, the full layer-by-layer spec:

> "the net contains eight layers with weights; the first five are
> convolutional and the remaining three are fully-connected. The output
> of the last fully-connected layer is fed to a 1000-way softmax... The
> first convolutional layer filters the 224x224x3 input image with 96
> kernels of size 11x11x3 with a stride of 4 pixels... The second
> convolutional layer takes as input the (response-normalized and pooled)
> output of the first convolutional layer and filters it with 256 kernels
> of size 5x5x48. The third, fourth, and fifth convolutional layers are
> connected to one another without any intervening pooling or
> normalization layers. The third convolutional layer has 384 kernels of
> size 3x3x256 connected to the (normalized, pooled) outputs of the
> second convolutional layer. The fourth convolutional layer has 384
> kernels of size 3x3x192, and the fifth convolutional layer has 256
> kernels of size 3x3x192. The fully-connected layers have 4096 neurons
> each."

(The "x192"/"x48" input-depth numbers come from the paper's two-GPU split,
Section 3.2 — each GPU holds half the kernels of a layer, with layer 3
being the one exception that reads from both GPUs' layer-2 output. Not
reproducing the two-GPU partition here, see entry 3.)

**ReLU** (Section 3.1), the paper's own first-listed and most-emphasized
contribution:

> "The standard way to model a neuron's output f as a function of its
> input x is with f(x) = tanh(x) or f(x) = (1+e^-x)^-1. In terms of
> training time with gradient descent, these saturating nonlinearities
> are much slower than the non-saturating nonlinearity f(x) = max(0,x)...
> Deep convolutional neural networks with ReLUs train several times
> faster than their equivalents with tanh units."

**Local Response Normalization** (Section 3.3), applied after ReLU in
layers 1 and 2 only:

> "b^i_{x,y} = a^i_{x,y} / (k + alpha * sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}
> (a^j_{x,y})^2)^beta ... k=2, n=5, alpha=1e-4, beta=0.75."

This normalizes each unit's activation against the same spatial position
in `n` neighboring *feature maps* (channels), not a spatial neighborhood —
easy to confuse with spatial pooling on a first read, since the formula
only has one spatial index `(x,y)` that stays fixed throughout.

**Overlapping max-pooling** (Section 3.4): pooling windows of size `z=3`,
stride `s=2` — i.e. stride smaller than the window, so adjacent pooling
regions overlap by one pixel. Follows both LRN layers and the 5th conv
layer.

**Weight init / training** (Section 5): weights from a zero-mean Gaussian
with std 0.01; biases initialized to 1 in conv layers 2/4/5 and both
hidden FC layers (to give ReLUs a positive input early on), 0 everywhere
else. SGD, batch 128, momentum 0.9, weight decay 0.0005, manually-stepped
LR starting at 0.01. **Dropout** (Section 4.2), rate 0.5, on the first two
FC layers only.

**Scale**: 60M parameters, 650K neurons, 1.2M training images at
224x224x3, trained for ~90 epochs over 5-6 days on two GTX 580 3GB GPUs.
Headline result: 37.5% top-1 / 17.0% top-5 error on ILSVRC-2010.

## Entry 3 — Scope: CIFAR-10, not ImageNet

Full ImageNet is explicitly out of reach here, and the paper says so
about itself: 5-6 days on two 2012-era GPUs, 1.2M images each needing
download+decode+resize, 60M parameters. None of that is this folder's
job to reproduce at that scale. Reproducing instead on **CIFAR-10**
(32x32x3, 10 classes, 50k/10k split) — the same substitution this
paper's own Section 3.1 makes for its ReLU-vs-tanh speed comparison
(Figure 1 is run on CIFAR-10, not ImageNet, "to make training as fast as
possible" — precedent set by the paper itself for exactly this kind of
smaller sanity-check experiment).

**What's reproduced, mechanism by mechanism, not just the headline
number**: ReLU (vs. a tanh baseline, reproducing the paper's own Figure 1
framing at small scale rather than just using ReLU uncritically), Local
Response Normalization (Eq. in entry 2, same k/n/alpha/beta constants),
overlapping max-pooling (z=3, s=2, vs. non-overlapping as an ablation,
reproducing the paper's own 0.4%/0.3% comparison framing from Section
3.4), and dropout on the FC layers (reusing 02_dropout's mechanism,
duplicated per this repo's self-contained-folder convention, not
imported).

**Re-scaled, with reasoning**: the literal kernel sizes (11x11 stride 4,
5x5, 3x3) are sized for a 224x224 input and would eat a 32x32 image in
one or two layers (an 11x11 stride-4 kernel alone leaves a 6x6 output).
Re-scaling kernel sizes/strides down for the 32x32 input while keeping
the paper's actual layer *count* (5 conv + 3 FC) and every mechanism
listed above is the only way to test "does ReLU+LRN+overlap-pool+dropout
help" rather than "did I pick input dimensions divisible by a stride of
4." Exact re-scaled kernel sizes are decided in the implementation entry
once the feature-map-size arithmetic is worked out, not guessed here.

**Substituted**: SGD+momentum+weight-decay (Section 5) is used as
specified, not replaced with Adam this time — unlike 06_lenet5, the
paper's own training recipe here is an ordinary, well-understood method
(not a bespoke second-order scheme like LeNet-5's Stochastic Diagonal
Levenberg-Marquardt), so there's no reason to substitute it.

**Not reproduced, with reasons**: the two-GPU split (Section 3.2) is a
1Z2012-hardware memory constraint, not an algorithmic contribution —
training on one array in one process is equivalent to the paper's own
stated "half as many kernels on one GPU" baseline comparison, which the
paper itself reports is 1.7%/1.2% worse, not a different algorithm. The
two forms of data augmentation (Section 4.1 — random 224x224 crops +
horizontal flips, and the PCA color-jitter scheme) are ImageNet-specific
preprocessing choices tied to the 256x256 source resolution; CIFAR-10's
own 32x32 images are already at the target resolution with no room for
that kind of crop augmentation at the same scale. ILSVRC-2012 multi-model
averaging results (Section 6, Table 2) are a competition-entry ensembling
detail, not part of the architecture.
