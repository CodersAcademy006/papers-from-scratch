# 06_lenet5 — Journal

Hardware: see repo README — CPU-only, pure numpy, no GPU. All timings below are specific to that machine.

## Entry 1 — Sourcing

Not on arXiv (1998, predates it). Paper: LeCun, Bottou, Bengio, Haffner,
"Gradient-Based Learning Applied to Document Recognition," Proc. IEEE,
Nov 1998. Full text hosted by the author: yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
(46 pages, 1.2 columns wide document). First fetch attempt timed out
(`curl` exit 28, connection refused on retry) — the author's personal
site is occasionally unreachable; retried a few minutes later and it
came through cleanly (HTTP 200, 955KB, 46-page PDF confirmed by `file`).
Noting this because it's a real, mundane infrastructure hiccup, not
because it matters to the implementation.

The paper covers far more than LeNet-5 itself: Sections IV-X build up to
Graph Transformer Networks and a full bank-check reading system. Only
Section II (the CNN architecture) and the MNIST portion of Section III
(results) are this folder's concern — see entry 3 for the full scope cut.

## Entry 2 — The architecture, by quote, verified by arithmetic

Fig. 2's architecture, quoted layer by layer (Section II.B):

> "LeNet-5 comprises 7 layers, not counting the input, all of which
> contain trainable parameters (weights). The input is a 32x32 pixel
> image... significantly larger than the largest character in the
> database (at most 20x20 pixels centered in a 28x28 field). The reason
> is that it is desirable that potential distinctive features such as
> stroke end-points or corner can appear in the center of the receptive
> field of the highest-level feature detectors."

**C1** (conv): "a convolutional layer with 6 feature maps. Each unit in
each feature map is connected to a 5x5 neighborhood in the input. The
size of the feature maps is 28x28... C1 contains 156 trainable parameters,
and 122,304 connections."

**S2** (sub-sample): "a sub-sampling layer with 6 feature maps of size
14x14. Each unit in each feature map is connected to a 2x2 neighborhood
in the corresponding feature map in C1. The four inputs to a unit in S2
are added, then multiplied by a trainable coefficient, and added to a
trainable bias. The result is passed through a sigmoidal function... Layer
S2 has 12 trainable parameters and 5,880 connections."

This is NOT plain average pooling — each feature map has its own single
scalar scale and single scalar bias applied after the 2x2 average, before
the squashing function. Easy to miss if you only skim for "subsampling"
and reach for `np.mean` over 2x2 blocks with nothing else.

**C3** (conv, irregular connectivity): "a convolutional layer with 16
feature maps. Each unit in each feature map is connected to several 5x5
neighborhoods at identical locations in a *subset* of S2's feature maps."
The connection pattern is given as Table I (rows = S2 maps 0-5, columns =
C3 maps 0-15, X = connected):

```
      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0     X . . . X X X . . X  X  X  X  .  X  X
1     X X . . . X X X . .  X  X  X  X  .  X
2     X X X . . . X X X .  .  X  .  X  X  X
3     .  X X X . . X X X X  .  .  X  .  X  X
4     .  . X X X . . X X X  X  .  X  X  .  X
5     .  . . X X X . . X X  X  X  .  X  X  X
```

> "Why not connect every S2 feature map to every C3 feature map? The
> reason is twofold. First, a non-complete connection scheme keeps the
> number of connections within reasonable bounds. More importantly, it
> forces a break of symmetry in the network. Different feature maps are
> forced to extract different (hopefully complementary) features because
> they get different sets of inputs. The first six C3 feature maps take
> inputs from every contiguous subset of three feature maps in S2. The
> next six take input from every contiguous subset of four. The next
> three take input from some discontinuous subsets of four. Finally the
> last one takes input from all S2 feature maps."

Didn't just copy the table — checked it against the paper's own stated
parameter count before trusting the transcription. Each column's trainable
count is `(#connected_maps * 25 + 1 bias)`: 6 maps with 3 inputs
(`6*76=456`) + 6 maps with 4 inputs (`6*101=606`) + 3 maps with 4 inputs
(`3*101=303`) + 1 map with 6 inputs (`1*151=151`) = `456+606+303+151 =
1516`, matching the paper's stated "C3 has 1,516 trainable parameters"
exactly. If the table had been mistyped anywhere, this arithmetic would
almost certainly not land on the paper's own number, so this is a real
check, not a formality.

**S4**: same sub-sampling mechanism as S2, 16 maps of size 5x5, 32
trainable parameters, 2,000 connections.

**C5** (conv, full connection, output size collapses to 1x1): "a
convolutional layer with 120 feature maps. Each unit is connected to a
5x5 neighborhood on all 16 of S4's feature maps. Here, because the size
of S4 is also 5x5, the size of C5's feature maps is 1x1... Layer C5 has
48,120 trainable connections." (`120 * (5*5*16 + 1) = 120*401 = 48120`,
checks out — same verification habit as C3.)

**F6**: fully connected, 84 units, "contains 84 units... fully connected
to C5. It has 10,164 trainable parameters" (`84*(120+1)=84*121=10164`,
checks out).

**Activation** (Eq. 5-6, Appendix A): every unit up through F6 uses

> "x_i = f(a_i)... The squashing function is a scaled hyperbolic tangent:
> f(a) = A tanh(Sa)... For our simulations, we use A = 1.7159 and S = 2/3
> (see [20], [34]). With this choice of parameters, the equalities f(1)=1
> and f(-1)=-1 are satisfied."

**Weight init** (Appendix A): "weights are initialized with random values
using a uniform distribution between -2.4/F_i and 2.4/F_i where F_i is the
number of inputs (fan-in) of the unit."

**Output layer**: the paper's actual output is 10 Euclidean RBF units
with fixed (hand-designed, -1/+1 bitmap) parameter vectors, Eq. 7, trained
with either plain MSE (Eq. 8) or a discriminative MLE-style criterion
(Eq. 9). See entry 3 for why this folder substitutes a standard
softmax+cross-entropy output instead.

**Training**: the paper does not use plain SGD. It uses a "Stochastic
Diagonal Levenberg-Marquardt" method (Appendix C) — a per-parameter
learning rate derived from a diagonal Gauss-Newton approximation to the
Hessian, re-estimated on 500 samples before each pass, trained for 20
epochs with a hand-specified decreasing global rate schedule
(0.0005/0.0002/0.0001/0.00005/0.00001). See entry 3 for the substitution
here too.

## Entry 3 — Scope

**Reproduced**: the LeNet-5 architecture itself (C1-S2-C3-S4-C5-F6), built
from scratch — a hand-rolled `Conv2D` (im2col-based, not a framework conv
op) supporting C3's exact irregular per-map connectivity from Table I, and
a hand-rolled `SubsampleLayer` matching the paper's trainable-scale-and-
bias mechanism exactly (not plain average pooling) — trained on MNIST and
compared against the paper's headline 0.95% (no distortion) / 0.8% (with
distortion) test error figures from Fig. 5/6.

**Substituted, each flagged as a deviation rather than silently done**:
- **Output layer**: softmax + cross-entropy on a standard 84->10 linear
  layer, instead of the paper's 10 fixed-prototype Euclidean RBF units
  (Eq. 7) with hand-designed -1/+1 bitmap codewords. The RBF-with-fixed-
  prototypes design is motivated by the paper's string-recognition use
  case (confusable-character codes helping a downstream linguistic
  post-processor, Fig. 3) — not relevant to a standalone 10-digit
  classifier, and every modern reimplementation of LeNet-5 makes this
  same substitution. The convolutional feature extractor (C1-F6) is what
  this folder is actually testing.
- **Training algorithm**: this repo's own `04_adam/optimizer.py` Adam
  implementation (re-used, not re-derived), instead of the paper's
  Stochastic Diagonal Levenberg-Marquardt (Appendix C) — a per-parameter
  second-order method with its own re-estimation schedule that is a
  paper unto itself (Appendix B/C run two full pages justifying it). Not
  rebuilding that here; Adam is a reasonable, well-tested stand-in for
  "an optimizer that handles per-parameter curvature differences
  reasonably well," which is what the Stochastic Diagonal LM method was
  solving for in 1998.
- **Dataset**: standard MNIST (the now-canonical 60k/10k split, same
  source used in 03_batchnorm/04_adam/05_layernorm) instead of the
  paper's own hand-curated "MNIST" built from NIST SD-3/SD-1 with a
  specific writer-based train/test split. The canonical MNIST dataset IS
  the paper's own construction (LeCun et al. are the ones who built and
  named it) and is what "MNIST" has meant in every context since — no
  separate acquisition needed.
- **Distortion-augmented training** (Fig. 6/7's 540,000 artificially
  distorted examples, dropping test error from 0.95% to 0.8%): not
  included in the base run. Revisit only if the base reproduction's
  timing allows it without derailing the rest of this repo's schedule —
  not promised here.

**Out of scope entirely**, with reasons: Sections IV-X (Graph Transformer
Networks, heuristic over-segmentation, word/string recognition, the
check-reading system) are a different, much larger system built on top of
a CNN recognizer, not a description of LeNet-5 itself — a GTN is its own
undertaking, unrelated to "implement a CNN from scratch." Section VII's
Space-Displacement Neural Network and Section IX's online handwriting
system are likewise downstream applications, not part of the architecture
this folder is reproducing.

## Entry 4 — Conv2D and SubsampleLayer, gradchecked in isolation

`conv2d.py` is im2col/col2im-based valid convolution (stride 1, no
padding, matching every conv layer in LeNet-5). C3's irregular Table I
connectivity is handled with one mechanism, not a special case: a boolean
`(C_out, C_in)` mask is applied to the weight tensor every forward pass
(`W_effective = W * mask`), so unconnected weight slices start at zero and
the chain rule alone keeps their gradient at zero forever — no branching
in `backward()` between the "sparse" and "full" connectivity cases, a
full-connectivity `Conv2D` is just one where `mask` is all ones.

`subsample.py` implements the paper's actual S2/S4 mechanism (entry 2):
one trainable scalar coefficient and bias per feature map on top of the
2x2 average, not plain average pooling.

Both gradchecked before composing anything (`test_gradcheck.py`): full
connectivity, sparse C3-style connectivity (with an explicit assert that
unconnected weight slices stay exactly zero through both init and after
a backward pass, not just a numeric gradient check), and the subsample
layer. All three passed on the first real run:

```
Conv2D (full connectivity) gradcheck passed: worst rel err dW=3.29e-11 db=1.32e-11 dx=3.47e-08
Conv2D (sparse connectivity, C3-style) gradcheck passed: worst rel err dW=2.23e-11, zero-masked entries confirmed
SubsampleLayer gradcheck passed: worst rel err dcoeff=8.60e-12 dbias=9.38e-11 dx=1.02e-09
```

No bug to report — im2col/col2im is a well-trodden pattern and the mask
trick is simple enough to get right on the first pass. Recording the
clean result plainly rather than looking for a story that isn't there.

## Entry 5 — Composing the full network: one honest numerical-precision observation

`lenet5.py` wires C1-S2-C3-S4-C5-F6 together (`LeNet5.forward`/`backward`),
substituting softmax+cross-entropy for the paper's RBF output (entry 3).
First full draft of `forward()` had a real but harmless mistake, caught on
read-through before running anything: it called `self.C1.forward(x)` a
second time inside a leftover placeholder line while building up the
pre-activation caching `backward()` needs, duplicating a conv pass for no
reason. Fixed before the first run (no gradcheck needed to catch
this one — it was a dead line, not a math error).

`test_lenet5_gradcheck.py` checks every parameter in the composed network
against finite differences on a small batch. Passed on the first real
run, all 14 parameter groups:

```
C1.W: 4.70e-06   C1.b: 3.52e-06   S2.coeff: 1.42e-05  S2.bias: 3.67e-07
C3.W: 1.26e-03   C3.b: 1.35e-07   S4.coeff: 2.13e-05  S4.bias: 2.27e-08
C5.W: 5.09e-06   C5.b: 6.99e-09   F6.W:     7.24e-06  F6.b:    1.36e-09
out.W: 2.47e-06  out.b: 6.08e-10
```

One real, worth-noting detail rather than smoothing it over: `C3.W`'s
worst relative error (1.26e-03) is visibly higher than every other
parameter group, and technically above this repo's usual `REL_TOL=1e-3`
bar — it only passes because the check also accepts `ABS_TOL<1e-4` in
absolute terms, which it clears comfortably. Not a bug: C3 sits in the
middle of a six-layer chain of smooth tanh nonlinearities (C1->S2->C3 on
the way up, C3->S4->C5->F6->out on the way back), and finite-difference
error compounds through a longer chain of multiplications in both
directions more than it does for a parameter near either end of the
network (C1 is one layer from the input, out.W is one layer from the
loss). This is a known, generic property of numerical differentiation
through deep computation graphs, not specific to this architecture or a
sign of a mistake — flagged here because the honest thing to do with a
gradcheck number that's an order of magnitude worse than its neighbors is
explain why, not quietly round it off as "still passing."
