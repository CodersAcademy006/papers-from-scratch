# Journal — 04_adam

Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015,
arXiv:1412.6980).

## Entry 1 — Sourcing

Direct arXiv full-text pull again, same as 03_batchnorm, no mirror fight.
Noting it anyway for the running record: 1 for 1 easy sourcing streak since
switching off papers old enough to predate arXiv (01_backprop_autograd was
the only hard one, being from 1986).

## Entry 2 — The algorithm, by quote

**The update rule itself (Algorithm 1)**, exactly as given, nothing
simplified:

```
m_0 = 0, v_0 = 0, t = 0
while not converged:
    t += 1
    g_t = grad_theta f_t(theta_{t-1})
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2          # elementwise square
    m_hat_t = m_t / (1 - beta1^t)
    v_hat_t = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + eps)
```

> "Good default settings for the tested machine learning problems are
> alpha = 0.001, beta1 = 0.9, beta2 = 0.999 and eps = 1e-8."

These defaults are used as-is below, not re-tuned, since the paper presents
them as defaults meant to transfer across problems rather than per-task
hyperparameters (unlike 02_dropout's `epsilon_0` or 03_batchnorm's
unstated MNIST learning rate, which genuinely had no paper-given value to
start from).

**Why the bias-correction terms exist at all** — this is the part most
blog-post implementations skip or get hand-wavy about, and the paper
derives it properly (Section 3), so it's transcribed here instead of
paraphrased:

> "v_t = (1 - beta2) * sum_{i=1}^t beta2^(t-i) * g_i^2 ... We wish to know
> how E[v_t]... relates to the true second moment E[g_t^2]... E[v_t] =
> E[g_t^2] * (1 - beta2^t) + zeta ... What is left is the term (1 -
> beta2^t) which is caused by initializing the running average with zeros.
> In algorithm 1 we therefore divide by this term to correct the
> initialization bias."

In words: an exponential moving average initialized at zero is biased
toward zero for exactly as long as the decay rate keeps "remembering" that
initial zero — dividing by `(1 - beta^t)` exactly cancels that geometric
factor. `t=1`, `beta2=0.999`: `1 - beta2^1 = 0.001`, so the very first
`v_hat_1` is `v_1 / 0.001`, a 1000x correction. The correction matters most
exactly when it's needed most (small `t`, `beta` close to 1) and fades to
~1 as `t` grows — not a one-off hack, a quantity derived to match the
actual size of the initialization artifact at each step.

**What "removing it" turns Adam into** — this motivates the ablation in
entry 4 below instead of requiring the paper's own VAE experiment:

> "Discussed in section 5, removal of the bias correction terms results in
> a version of RMSProp (Tieleman & Hinton, 2012) with momentum... RMSProp
> also lacks a bias-correction term; this matters most in case of a value
> of beta2 close to 1 (required in case of sparse gradients), since in that
> case not correcting the bias leads to very large stepsizes and often
> divergence, as we also empirically demonstrate in section 6.4."

**The "trust region" property (Section 2.1)**, worth quoting because it's
the clearest statement of why Adam's step sizes behave the way they do, and
gives a concrete numeric claim to test in entry 5:

> "The effective magnitude of the steps taken in parameter space at each
> timestep are approximately bounded by the stepsize setting alpha...
> rescaling the gradients g with factor c will scale m_hat with a factor c
> and v_hat with a factor c^2, which cancel out."

**The AdaGrad correspondence (Section 5)** — a concrete, checkable
mathematical claim, not just a qualitative comparison:

> "AdaGrad corresponds to a version of Adam with beta1 = 0, infinitesimal
> (1 - beta2) and a replacement of alpha by an annealed version alpha_t =
> alpha * t^(-1/2)."

This is directly testable: run Adam with `beta1=0`, `beta2` extremely
close to 1, `alpha_t = alpha/sqrt(t)`, against a literal AdaGrad
implementation (`theta -= alpha * g_t / sqrt(sum_{i<=t} g_i^2)`) on the same
synthetic gradient sequence, and check the two update sequences actually
converge to each other numerically — see entry 6.

## Entry 3 — Scope decision

The paper runs four experiments (6.1 logistic regression, 6.2 multi-layer
NN, 6.3 CNN, 6.4 bias-correction on a VAE) plus a convergence-rate proof
(Section 4, pure online-convex-optimization theory, not something to
"implement," skipped entirely — there's no code to write for a regret
bound).

**Reproduced**: Section 6.2's deterministic multi-layer network setup:

> "a neural network model with two fully connected hidden layers with 1000
> hidden units each and ReLU activation... minibatch size of 128... the
> standard deterministic cross-entropy objective function with L2 weight
> decay."

On MNIST, comparing Adam against plain momentum-SGD and AdaGrad (both
cheap to add) at the same architecture/init — the dropout variant of this
same experiment and the SFO quasi-Newton baseline are both skipped (SFO
specifically because it's "5-10x slower per iteration" even in the paper's
own words and has its own memory/implementation complexity that's a
separate undertaking; dropout variant skipped because 02_dropout already
covers dropout's own mechanism in depth and re-adding it here would just
be testing the same mechanism twice rather than testing Adam).

**Reproduced, but on a cheaper setup than the paper's own**: Section 6.4's
bias-correction claim. The paper tests this on a VAE with a 50-dimensional
latent Gaussian (Kingma & Welling, 2013) — implementing a VAE is paper
28_vae's job, not this one's, so instead the *claim itself* ("beta2 close
to 1 leads to instabilities... when no bias correction term was present,
especially at first few epochs") is tested directly on 04_adam's own
Section 6.2 network: same architecture, same data, Adam with vs. without
the bias-correction division, at `beta2=0.999`. If the underlying
mechanism is what the paper says it is, the effect should show up on any
network trained with a high `beta2`, not just a VAE — this is a case of
testing the mechanism rather than literally reproducing the figure.

**Reproduced as an extra, paper-doesn't-ask-for-it check**: the AdaGrad
correspondence from entry 2, verified numerically on a synthetic gradient
sequence rather than a real network (it's a claim about the optimizers'
update rules in the abstract, doesn't need training data to test).

**Out of scope, with reasons**: 6.1 logistic regression + IMDB sparse
bag-of-words experiment (a second, different dataset/task whose main point
-- sparse gradient handling -- is a repeat of the same adaptive-learning-
rate idea already covered by the AdaGrad correspondence check); 6.3 CNN
experiment (needs a hand-rolled conv2d, which is 06_lenet5 and 07_alexnet's
job, not this optimizer paper's); Section 4's regret-bound proof and
AdaMax/temporal-averaging extensions (Sections 7.1, 7.2) -- interesting,
but they're each a distinct secondary mechanism layered on top of the
paper's actual contribution, not needed to validate Adam itself.
