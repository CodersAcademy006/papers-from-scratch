# Journal — 03_batchnorm

Paper: "Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift" (Ioffe & Szegedy, arXiv:1502.03167, 2015).

## Entry 1 — Sourcing

On arXiv, full text pulled directly through `mcp__alphaxiv__get_paper_content`
(fullText=true) in one call, equations and all. No PDF-mirror fight this time
(contrast with 01_backprop_autograd, which predates arXiv and took two mirror
attempts; and 02_dropout, which needed the same tool but was otherwise easy).
Worth noting only because it won't always be this easy — 01 and 02 both took
real effort just to get a working text source.

## Entry 2 — What the paper actually specifies, by quote

Quoting the paper directly for every design decision below, not paraphrasing,
so a reader can check the implementation against the paper's own words
instead of my summary of them.

**The core transform (Algorithm 1).** Per-feature, not joint/whitened:

> "instead of whitening the features in layer inputs and outputs jointly, we
> will normalize each scalar feature independently, by making it have the
> mean of zero and the variance of 1."

Exact steps, batch B of size m, one scalar feature at a time:

```
mu_B    = (1/m) * sum_i x_i
sigma_B^2 = (1/m) * sum_i (x_i - mu_B)^2
x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + eps)
y_i     = gamma * x_hat_i + beta  =  BN_{gamma,beta}(x_i)
```

**Why gamma/beta exist at all** (not just normalize and stop) — restoring
representational power:

> "normalizing the inputs of a sigmoid would constrain them to the linear
> regime of the nonlinearity. To address this, we make sure that the
> transformation inserted in the network can represent the identity
> transform... by setting gamma^(k) = sqrt(Var[x^(k)]) and beta^(k) =
> E[x^(k)], we could recover the original activations, if that were the
> optimal thing to do."

**Where exactly BN goes relative to the nonlinearity** — before it, on
`Wu+b`, not on the layer's raw input `u`:

> "We add the BN transform immediately before the nonlinearity, by
> normalizing x = Wu + b... since u is likely the output of another
> nonlinearity, the shape of its distribution is likely to change during
> training, and constraining its first and second moments would not
> eliminate the covariate shift. In contrast, Wu + b is more likely to have
> a symmetric, non-sparse distribution."

And a detail that's easy to miss and silently get wrong: the bias `b` becomes
redundant once BN subtracts the batch mean, because any constant shift from
`b` is exactly cancelled by `mu_B` subtraction:

> "since we normalize Wu+b, the bias b can be ignored since its effect will
> be canceled by the subsequent mean subtraction (the role of the bias is
> subsumed by beta in Alg. 1)."

So the implementation below has NO bias term on the pre-BN linear layer —
only `W`, with `beta` doing the job a bias normally would. This is a
one-line omission that is very easy to get wrong by habit (every linear
layer "should" have a bias) and the paper calls it out explicitly for
exactly that reason.

**Train vs. inference — the part most tutorials gloss over** (Algorithm 2).
At train time, mean/var come from the mini-batch and are part of the
computation graph. At inference, they're frozen population statistics:

> "The normalization of activations that depends on the mini-batch allows
> efficient training, but is neither necessary nor desirable during
> inference; we want the output to depend only on the input,
> deterministically... we use the normalization x_hat = (x - E[x]) /
> sqrt(Var[x] + eps) using the population, rather than mini-batch,
> statistics... We use the unbiased variance estimate Var[x] = m/(m-1) *
> E_B[sigma_B^2], where the expectation is over training mini-batches."

The paper is deliberately vague on *how* to accumulate `E_B[mu_B]` and
`E_B[sigma_B^2]` across many mini-batches ("process multiple training
mini-batches... and average over them") — it does not specify a running
exponential average, which is what basically every modern framework (and
what BatchNorm became known for) actually does. Implemented here as a
genuinely literal running (arithmetic, not exponential) average over every
mini-batch seen during training, specifically because that's the literal
reading of "average over them" — and flagged here as a deliberate deviation
from the EMA convention that came later, same spirit as 02_dropout's choice
to implement the paper's actual historical "halve the weights" mean-network
instead of the inverted-dropout convention that superseded it.

**Backward pass — given to us directly, not left as an exercise.** The paper
hands over the exact chain-rule decomposition (Section 3, "before
simplification"):

```
dL/dx_hat_i = dL/dy_i * gamma
dL/dsigma_B^2 = sum_i dL/dx_hat_i * (x_i - mu_B) * (-1/2) * (sigma_B^2+eps)^(-3/2)
dL/dmu_B = (sum_i dL/dx_hat_i * -1/sqrt(sigma_B^2+eps)) + dL/dsigma_B^2 * sum_i(-2(x_i-mu_B))/m
dL/dx_i = dL/dx_hat_i * 1/sqrt(sigma_B^2+eps) + dL/dsigma_B^2 * 2(x_i-mu_B)/m + dL/dmu_B * 1/m
dL/dgamma = sum_i dL/dy_i * x_hat_i
dL/dbeta  = sum_i dL/dy_i
```

This is implemented literally, term by term, in `bn_layer.py`, rather than
using the single-pass algebraically-simplified form that's common in later
blog posts/frameworks — closer to the paper's own derivation, and much
easier to cross-check line-by-line against the quote above.

**The warning the paper gives about getting this wrong** — this became the
most useful part of the whole paper for this exercise, see Entry 4:

> "consider a layer with the input u that adds the learned bias b, and
> normalizes the result by subtracting the mean of the activation computed
> over the training data... If a gradient descent step ignores the
> dependence of E[x] on b, then it will update b <- b + db... Thus, the
> combination of the update to b and subsequent change in normalization led
> to no change in the output of the layer nor, consequently, the loss. As
> the training continues, b will grow indefinitely while the loss remains
> fixed... We have observed this empirically in initial experiments, where
> the model blows up when the normalization parameters are computed outside
> the gradient descent step."

The paper is describing exactly the bug of computing gradients as if
`mu_B`/`sigma_B^2` were constants with respect to `x`, instead of
differentiating through them. Section 4.

## Entry 3 — Scope decision

Unlike 01 and 02, the paper's own validating experiment for the mechanism
itself (not the full ImageNet/Inception result) is small enough to run
exactly as described, on a laptop, in pure numpy. Section 4.1:

> "we considered the problem of predicting the digit class on the MNIST
> dataset... a very simple network, with a 28x28 binary image as input, and
> 3 fully-connected hidden layers with 100 activations each. Each hidden
> layer computes y = g(Wu+b) with sigmoid nonlinearity... followed by a
> fully-connected layer with 10 activations (one per class) and
> cross-entropy loss. We trained the network for 50000 steps, with 60
> examples per mini-batch."

This is being reproduced exactly: 784-100-100-100-10, sigmoid hidden units,
cross-entropy + softmax output, 50000 SGD steps, batch size 60, BN inserted
before every hidden sigmoid (per Entry 2's "immediately before the
nonlinearity" quote). The paper does not state a learning rate for this
specific toy experiment (it only gives learning rates for the much larger
Inception/ImageNet experiment in 4.2, which is explicitly a different,
much bigger network with momentum SGD — not reproduced here, out of scope,
same reasoning as skipping ImageNet in every prior paper in this repo: it
needs a distributed cluster, not a laptop). A fixed learning rate was
chosen empirically for this toy network (documented with the actual search
in Entry 5) since the paper leaves it unspecified for this exact setup.

Two things are reproduced from Section 4, beyond the headline "BN trains
faster":
1. Fig. 1(a): test accuracy vs. training step, with vs. without BN, same
   architecture, same learning rate.
2. Fig. 1(b,c): the {15,50,85}th-percentile evolution of a sigmoid's input
   distribution over training — the actual internal-covariate-shift
   evidence, not just the end accuracy number.
3. An extra ablation not in the paper's MNIST section but directly testing
   its Section 3.3 claim ("Batch Normalization allows us to use much higher
   learning rates... without the risk of divergence"): run both networks at
   a deliberately large learning rate and check which one survives.

Out of scope, same as every other paper here: the Inception/ImageNet
experiment (13.6M-parameter conv net, distributed training, ensemble of 6
networks) — a multi-day, multi-GPU undertaking, not attempted.
