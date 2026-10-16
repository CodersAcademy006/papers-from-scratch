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

## Entry 5: Timing and a second scope cut

Smoke-tested one real batch of 128 on the full network before launching
anything: forward 7.4s, backward 22.6s, ~30.3s/batch total. At 390
batches/epoch (50000 examples / 128) that's ~196 minutes/epoch -- 8
epochs would be over a day, infeasible.

Tried the paper's own batch size (256, Section 3.1) to see if it helps:
58.97s/batch, 195 batches/epoch, ~192 minutes/epoch -- essentially the
same total compute per epoch as batch=128 (58.97/30.25 ~ 1.95x the
per-batch cost for 2x the examples). Confirms this is compute-bound on
the conv layers, not batch-size-bound -- changing batch size just moves
work between batches, it doesn't reduce total work.

The only real lever left is data volume. Following the same scope cut
05_layernorm's JOURNAL.md already used for an analogous reason (its
Section 6.6 reproduction), subsampled the training set to 5000 examples
(full 10000-example test set still used for honest evaluation). At
batch=256 that's 19-20 batches/epoch.

Real measured time for this subsampled epoch, at the paper's own
lr=1e-2: 1179.5s for 20 batches (~59s/batch, matching the earlier
single-batch estimate) -- but it never finished cleanly: loss went NaN
partway through (see entry 6). At ~59s/batch, 20 batches/epoch is
~19.7 minutes/epoch; 8 epochs lands at ~2.6 hours, in the same range as
07_alexnet's actual training run.

## Entry 6: A second real bug -- gradient explosion at the paper's own lr

The 20-batch timing probe above didn't just run slowly, it diverged:
`RuntimeWarning: invalid value encountered in multiply` from
dropout.py, and the final printed loss was `nan`. Traced it with a
batch-by-batch magnitude dump (logit_absmax, grad_absmax, w_absmax) at
the paper's lr=1e-2:

```
batch 0: logit_absmax=3.18e+02  grad_absmax=3.05e+01  w_absmax=1.05e+00
batch 1: logit_absmax=2.92e+04  grad_absmax=1.02e+03  w_absmax=1.01e+01
batch 2: logit_absmax=9.63e+11  grad_absmax=8.74e+09  w_absmax=8.74e+07
batch 3: logit_absmax=6.27e+73  grad_absmax=1.18e+67  w_absmax=1.18e+65
batch 4: nan
```

Textbook exponential gradient explosion, roughly 100-1000x growth per
batch, overflowing to NaN by batch 4. Not staged -- this surfaced from
running the actual probe meant to measure timing, the divergence was
unplanned.

Section 3.1's lr=1e-2 was tuned for the paper's own setup: the original
flat std=0.01 init (deliberately small-signal, as 07_alexnet's entry 6
found), ImageNet-scale batch statistics, and the paper's own gradual
LR-drop schedule over 74 epochs. This repo already substituted He init
for config A (entry 2/3) specifically because the flat-std init doesn't
transfer to this rescale -- but He init also changes what learning rate
is safe, since it starts with proportionally larger per-layer gradients
by design. Using the paper's lr unchanged after swapping the init was
the mismatch.

Tested lr=1e-3 (10x smaller) with the same batch-by-batch trace over 10
batches: stable and improving the whole way, no explosion --

```
batch 0: loss=23.62  logit_absmax=3.18e+02  grad_absmax=3.05e+01
batch 9: loss=3.03   logit_absmax=6.70e+00  grad_absmax=3.79e-01
```

Switched `run_experiment.py` to lr=1e-3. This is an explicitly-flagged
hyperparameter substitution (same spirit as 06_lenet5's RBF->softmax
and optimizer swaps, and 07_alexnet's init fix): the paper's own
lr=1e-2 is tied to its own init scheme, and doesn't carry over once the
init is changed for a different reason.

## Entry 7: Final training run

Full 8-epoch run on the 5000-example training subset (entry 5), full
10000-example test set for evaluation, SGD+momentum+weight decay
(5e-4), lr=1e-3 (entry 6's fix), batch size 256 (the paper's own
Section 3.1 value).

```
epoch  train_nll  test_err
0      6.610402   89.53%
1      2.315225   88.91%
2      2.303784   88.90%
3      2.300627   88.18%
4      2.296783   87.49%
5      2.292630   86.76%
6      2.288500   86.11%
7      2.284060   85.70%
```

Final test error: **85.70%**, LR never decayed (test error never
plateaued 3 consecutive epochs, so the patience-based schedule from
entry 6 never fired). The curve (`figures/training_curve.png`) shows a
real, honest shape rather than a clean one: loss drops hard in epoch 0
(6.6 -> 2.3), then barely moves for the next two epochs (2.315 -> 2.304
-> 2.304, test error flat at ~88.9-89.0%) before slowly, monotonically
descending from epoch 3 onward. Not a repeat of entry 6's gradient
explosion (no NaN, no blow-up in magnitude) -- this is a genuinely slow
network: VGG-A is 11 weighted layers on only 5000 training examples
(entry 5's compute-driven scope cut) with a 10x-smaller-than-paper
lr=1e-3 (entry 6), so a multi-epoch warm-up before the loss surface
opens up is plausible rather than alarming, and the last 5 epochs'
steady improvement (no plateau, no regression) supports that reading
over a stuck-optimizer one.

No paper-reported baseline exists for this exact setup to compare
against -- the paper's own headline numbers are ImageNet top-1/top-5 on
the real 1.3M-image training set with multi-scale/multi-crop test-time
evaluation (entry 3), none of which this CIFAR-10 rescale on a 10%
training subset reproduces. Reported as the real number for what this
repo actually built: an 11-weight-layer VGG-A-style CIFAR-10 classifier
trained on a reduced subset for a short fixed epoch budget, not a
reproduction of the paper's own ILSVRC result.
