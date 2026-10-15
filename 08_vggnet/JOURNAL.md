# VGGNet (Very Deep Convolutional Networks for Large-Scale Image Recognition)

Hardware: see repo README -- CPU-only, pure numpy, no GPU. All timings below are specific to that machine.

## Entry 1: Sourcing

Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
Image Recognition", ICLR 2015. arXiv:1409.1556.

Fetched cleanly on the first try: `curl -sL -f https://arxiv.org/pdf/1409.1556`
-> 200 OK, 200010 bytes, 16 pages (main paper; appendices follow).

## Entry 2: Architecture (Section 2, Table 1)

Generic layout (Section 2.1): fixed 224x224 RGB input, only preprocessing
is "subtracting the mean RGB value, computed on the training set, from
each pixel." All conv layers use 3x3 filters (the paper's central claim:
stacking small filters instead of using the larger receptive fields of
prior architectures), stride 1, padding 1 ("same" -- spatial resolution
is preserved through each conv). Five max-pool layers (2x2 window,
stride 2) follow some (not all) of the conv layers. ReLU after every
hidden layer. Channel width starts at 64 and doubles after each
max-pool, up to 512.

Table 1 gives six configurations (A through E, 11 to 19 weight layers).
This repo implements **configuration A** (11 weight layers: 8 conv + 3
FC) -- the shallowest, quoted exactly from Table 1:

```
conv3-64
maxpool
conv3-128
maxpool
conv3-256, conv3-256
maxpool
conv3-512, conv3-512
maxpool
conv3-512, conv3-512
maxpool
FC-4096, FC-4096, FC-1000, softmax
```

None of the networks use Local Response Normalisation except the A-LRN
variant (Section 2.1: "such normalisation does not improve performance
... but leads to increased memory consumption and computation time") --
config A has no LRN, so this implementation has none either, unlike
07_alexnet.

Training (Section 3.1): multinomial logistic regression (softmax + CE),
mini-batch SGD with momentum 0.9, batch size 256, weight decay (L2,
multiplier 5e-4), dropout 0.5 on the first two FC layers. LR starts at
1e-2, divided by 10 when validation accuracy stops improving, decayed 3
times total over 74 epochs (370K iterations) in the paper's own run.

Weight init (Section 3.1): "we began with training the configuration A
... random initialisation. Then, when training deeper architectures, we
initialised the first four convolutional layers and the last three
fully-connected layers with the layers of net A." Since this repo only
implements config A itself, that pretraining-chain doesn't apply here --
A is the base case and is randomly initialised throughout. The paper's
own random-init recipe: "weights sampled from a normal distribution with
zero mean and 1e-2 variance [i.e. std=0.01]... biases initialised with
zero," with a footnote that Glorot & Bengio (2010) random init also
works without any pretraining.

Note on that std=0.01 figure: 07_alexnet/JOURNAL.md entry 6 found this
exact flat-std-Gaussian init (also a Krizhevsky et al. 2012 convention,
which this paper's own init recipe explicitly follows) silently kills
training on a channel/depth-rescaled network -- activation variance
compounds downward across every layer until the input signal is buried.
Config A is 11 weight layers deep (3 more than AlexNetCIFAR's 8), so the
same failure mode is expected here too. This repo uses He init
(std=sqrt(2/fan_in), bias=0) from the start instead of reproducing that
known failure -- see entry 3.

## Entry 3: Scope for a CIFAR-10 rescale

Same reasoning as 07_alexnet: full ImageNet (224x224, 1000-way, 1.3M
training images, 2-3 weeks on four Titan Black GPUs per the paper's own
Section 3.3) is infeasible on this machine. CIFAR-10 (32x32x3, 10
classes) is used instead.

Unlike AlexNet's rescale, VGG's architecture maps onto 32x32 input
cleanly without changing channel counts or dropping max-pool stages:
five 2x2/stride-2 max-pools take 32 -> 16 -> 8 -> 4 -> 2 -> 1, landing
exactly at a 1x1 spatial map (vs. the paper's 7x7 at 224x224 input) just
like the paper's own five-maxpool design intends. So config A's channel
schedule (64, 128, 256, 256, 512, 512, 512, 512) is used verbatim, not
scaled down -- the only architectural change is the flattened FC6 input
width (512*1*1=512 here vs. the paper's 512*7*7=25088) and the final
layer (FC-10 softmax instead of FC-1000).

What's reproduced mechanism-by-mechanism: the conv3-everywhere design,
5 max-pools doubling channels, ReLU, dropout 0.5 on the first two FC
layers, SGD with momentum + weight decay (same optimizer this repo
already built for 07_alexnet, reused here since VGG's Section 3.1
recipe is effectively the same SGD+momentum+decay scheme AlexNet used).

What's substituted: He init instead of the paper's flat std=0.01
Gaussian (entry 2's reasoning). Softmax+CE for 10 classes instead of
1000 -- this isn't really a substitution, the paper already uses
softmax+CE natively (unlike AlexNet's RBF-vs-softmax situation in
06_lenet5), just at a different class count.

What's NOT reproduced: multi-scale training/testing (Section 3.1's S in
[256,512] jitter and Section 3.2's dense/multi-crop evaluation) -- tied
to ImageNet's much larger native resolution, no clean analog at CIFAR's
32x32; the pretraining-chain init strategy for deeper configs (B-E),
since only config A is built; horizontal-flip+colour-jitter augmentation
(same reasoning as 07_alexnet's scope cut); the localisation task
(Appendix A) and multi-dataset generalisation study (Appendix B) --
different systems built on top of the classification net, out of scope.

## Entry 4: Building and gradchecking

Built `conv2d.py` and `maxpool.py` (duplicated from 07_alexnet, maxpool
re-parametrised to VGG's non-overlapping 2x2/stride-2 window instead of
AlexNet's overlapping 3x3/stride-2 -- same backward logic, different
(z, s) at construction). `dense.py` and `dropout.py` duplicated
unchanged. Gradchecked Conv2D (stride=1, padding=1, VGG's only
configuration) and MaxPool in isolation first (`test_gradcheck.py`):
both passed clean on the first real run (dx/dW/db worst 4.83e-09, dx
worst 2.27e-10) -- genuinely, no staged bugs.

Composed `vgg_cifar.py` (`VGG_A_CIFAR`): 8 conv layers across 5
conv-pool stages (1, 1, 2, 2, 2 convs per stage matching config A) then
FC6(512->4096)-FC7(4096->4096)-FC8(4096->10), dropout on FC6/FC7. Wrote
`test_vgg_gradcheck.py` for the full composed network.

The first full gradcheck run found real finite-difference measurement
trouble, not a bug: at the same EPS_FD=1e-5 used for 07_alexnet's
end-to-end check, conv1_1.b failed (rel err 3.3e-3); tightening to
1e-7 fixed that parameter but then conv2_1.b failed at a *larger*
relative error (3.6e-2) than before. Checked both by sweeping EPS_FD
from 1e-3 down to 1e-8 directly: each parameter's numeric gradient
converges exactly to its analytic value, but at a different "sweet
spot" eps per parameter (conv1_1.b converged by 1e-7, conv2_1.b needed
1e-8) -- central differencing has both O(eps^2) truncation error at
large eps and O(machine-eps/eps) round-off error at small eps, and
which one dominates for a given parameter depends on where it sits in
an 11-weight-layer chain of ReLU/maxpool nonlinearities. This is the
same phenomenon behind 06_lenet5's C3.W and 07_alexnet's conv1.W/
conv3.W gradcheck findings, just more pronounced here because config A
is deeper than either of those networks.

Fixed by setting EPS_FD=1e-8 for the full-network check (isolated
single-layer checks in test_gradcheck.py didn't need this -- only the
long composed chain did). At that step size all 16 parameter groups
pass cleanly, worst relative error 3.35e-04, well under REL_TOL=1e-3 --
confirmed stable across repeated runs.
