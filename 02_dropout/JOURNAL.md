# Journal — 02: Improving Neural Networks by Preventing Co-adaptation of Feature Detectors

Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov. arXiv:1207.0580 (2012). This is the short, original dropout report, not the longer 2014 JMLR paper — the README links the 2012 one specifically.

## Entry 1 — Sourcing

On arXiv, so this one was easy: fetched directly via alphaxiv's `get_paper_content` with `fullText=true`, full 18-page text (including all appendices) in one call, no mirror hunting needed this time (contrast with paper 01's three-hop detour through a bot wall and a WebFetch PDF-decoding failure).

## Entry 2 — The actual mechanism, as the paper states it

Core idea, stated in the abstract and intro: "on each presentation of each training case, each hidden unit is randomly omitted from the network with a probability of 0.5, so a hidden unit cannot rely on other hidden units being present."

- **Training:** for each minibatch, sample a fresh Bernoulli(0.5) mask per hidden unit per layer, zero out the masked units' activations, do forward+backward as normal through the surviving subnetwork, update only the weights touching surviving units (weights for dropped units simply get zero gradient that step, since their activation was zero).
- **Test time:** "we use the 'mean network' that contains all of the hidden units but with their outgoing weights halved to compensate for the fact that twice as many of them are active." This is the specific trick this paper uses — halve outgoing weights once, at test time, **not** the later "inverted dropout" convention (scale activations by `1/keep_prob` during training instead) that became standard afterward. Implementing the paper's actual convention, not the modern one, since the point is fidelity to this specific paper.
- **Input dropout:** also drops a random 20% of raw input pixels in several experiments (MNIST, TIMIT), on top of 50% hidden dropout — stated as "dropping out a random 20% of the pixels," i.e. retain probability 0.8 for inputs vs. 0.5 for hidden units.
- **Max-norm weight constraint**, used instead of an L2 penalty: "we set an upper bound on the L2 norm of the incoming weight vector for each individual hidden unit. If a weight-update violates this constraint, we renormalize the weights of the hidden unit by division." Appendix A.1 gives the concrete value for MNIST: squared-length bound `l = 15`, found by cross-validation.
- **Training schedule (Appendix A.1, MNIST), transcribed exactly:**

```
delta_w_t = p_t * delta_w_{t-1} - (1 - p_t) * epsilon_t * <grad_w L>_t
w_t = w_{t-1} + delta_w_t

epsilon_t = epsilon_0 * f^t
p_t = (t/T) * p_i + (1 - t/T) * p_f      for t < T
p_t = p_f                                 for t >= T

epsilon_0 = 10.0, f = 0.998, p_i = 0.5, p_f = 0.99, T = 500
```

Worth stating plainly how this compares to paper 01's momentum (eq. 9 there: `delta_w(t) = -eps*dE/dw(t) + alpha*delta_w(t-1)`, constant `alpha`): this paper's version is an exponential-moving-average form where the **new gradient term is also scaled by `(1 - p_t)`**, not just by the learning rate, and the momentum coefficient itself `p_t` is scheduled, ramping linearly from 0.5 to 0.99 over the first 500 epochs rather than held fixed. Two different parameterizations of "momentum," both legitimate, from the same research group six years apart.

- Minibatch size 100, weights initialized from a zero-mean normal with std 0.01, no weight decay, 3000 epochs, architectures like 784-800-800-10.

## Entry 3 — Scope decision: what gets reproduced here, and why

The paper's headline MNIST numbers (784-800-800-10 and similar, 60,000 training images, 3000 epochs, minibatch 100) are **not computationally reachable** with a pure-numpy, no-GPU, from-scratch implementation in a reasonable amount of time for this exercise: back-of-envelope, a single 784-800-800-10 forward+backward on 60,000 examples in minibatches of 100 for 3000 epochs is on the order of 10^14-10^15 floating point operations total in naive dense numpy — plausibly hours to tens of hours on a laptop CPU with no batching tricks beyond what numpy already vectorizes. Spending that much wall-clock time to reproduce one paper's exact error count, in a repo meant to cover many papers, is the wrong trade.

Decision: reproduce the **mechanism** exactly as specified (Bernoulli masking, mean-network test-time weight-halving, max-norm constraint, the scheduled-momentum update rule above, matching hyperparameter values where feasible) and the **qualitative claim the paper is actually about — overfitting reduction when a network is trained on a small dataset** — rather than the exact headline error counts, which would require the full-scale, full-duration run. This is directly licensed by the paper's own framing, not a dodge: the abstract's own motivating scenario is *"when a large feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data"* — so training on a deliberately small subset of MNIST is testing exactly the regime the paper itself opens with, just at a scale that finishes in minutes instead of hours.

Concrete plan: 784-256-128-10 network (smaller than the paper's 800/800, chosen for CPU runtime, documented as a deviation), trained on a **2,000-image subset** of MNIST (small enough to overfit badly without dropout, per the paper's own stated failure mode), evaluated on the full 10,000-image test set, for a few hundred epochs rather than 3000 — again a documented deviation, not a hidden one. Two runs with everything else held identical: (a) no dropout, max-norm and the scheduled-momentum optimizer still applied to both so the *only* difference between the two runs is dropout itself, which is a cleaner ablation than the paper's own comparison (the paper's "best published result... for a standard feedforward neural network" baseline of 160 errors was produced by an entirely different, unspecified training setup, not literally the same code with dropout switched off) — worth noting as a methodological improvement over the paper's own comparison, not just a limitation.

Not reproduced at all: TIMIT, CIFAR-10, ImageNet, Reuters experiments (Appendices B-E) and the convolutional-network architecture details (Appendix F-H) — these depend on datasets, a decoder (Viterbi/HMM for TIMIT), and GPU-scale convolutional training that are each their own multi-day undertaking, not a variation on the same core mechanism. The MNIST result is the one fully self-contained, CPU-tractable, numerically checkable experiment in the paper.

## Entry 4 — MNIST download broke, fixed, moving on to the model

`mnist_data.py` fetches the original idx-format files from a public S3 mirror (`ossci-datasets.s3.amazonaws.com`, the same mirror torchvision uses) via stdlib `urllib`, to avoid adding a dependency just to pull a dataset.

First run failed: `ssl.SSLCertVerificationError: unable to get local issuer certificate`. This is the python.org macOS framework-build Python not trusting this host's certificate chain, even though plain `curl` to the same URL works fine from the same shell (confirmed with `curl -sI` before touching the loader code at all) — a local root-cert-store gap in this Python install, not a problem with the host or the data. Fix: shell out to `curl` via `subprocess.run` instead of `urllib.request.urlretrieve`, rather than pulling in `certifi` as a new dependency just to fetch one dataset once. Re-ran, worked immediately: train `(60000, 784)`/`(60000,)`, test `(10000, 784)`/`(10000,)`, pixel values correctly normalized to `[0.0, 1.0]`, labels `0-9`.

## Entry 5 — Model, and a gradient-check false alarm

`mlp.py`: `DropoutMLP`, sigmoid hidden units (activation function not actually specified by the paper's main text for this plain-MLP experiment, only for the CIFAR/ImageNet conv nets in Appendix F where it's explicitly ReLU — filling the gap with sigmoid to match the era and paper 01's architecture, documented here rather than silently assumed), softmax output, cross-entropy loss (the loss Appendix A.1 does name explicitly). Implements: Bernoulli dropout masks at train time (separate mask per dropped layer, held fixed for one forward+backward pass), the paper's own test-time "mean network" (scale each weight matrix by the keep-probability of its *sending* layer, not the receiving one — the detail that needs to be gotten right here, since it is a per-weight-matrix scale tied to the layer the weights come *from*), per-hidden-unit max-norm weight constraint applied after every update, and the exact scheduled-momentum rule from Appendix A.1 (`delta_w_t = p_t*delta_w_{t-1} - (1-p_t)*epsilon_t*grad`).

`test_gradcheck.py` reuses 01_backprop_autograd's finite-difference approach but has to handle something that paper didn't need to: dropout masks must be held **fixed** across the `+eps`/`-eps` perturbations used to estimate each numerical gradient, or the comparison is meaningless (a fresh random mask each call means the loss depends on the mask draw too, not just the perturbed weight).

First run failed:
```
AssertionError: layer 0 W(1, 9): analytic=-0.00000005 numeric=-0.00000005 rel_err=1.01e-03
```
Both values print identically at 8 decimal places, yet the check reported a 1000x-tolerance relative error. The parameter sampled happened to belong to a hidden unit that was dropped (masked to zero) in the one forward pass used for this check, so its true gradient is genuinely ~0 on both sides — and pure relative error (`|num-ana| / max(|num|,|ana|)`) is numerically unstable exactly when both quantities are near zero, regardless of whether the implementation is correct. This is not a bug in `backward()`; it's a bug in the *check itself* for near-zero gradients. Fixed by switching to a combined absolute+relative tolerance (`np.isclose(num, ana, rtol=1e-4, atol=1e-6)`, the same two-term criterion `numpy.allclose` uses internally) instead of pure relative error. Re-ran: **passed, 24 sampled parameters across 3 layers, worst relative error 1.78e-03** (the same near-zero case as before, now correctly accepted since its absolute difference is ~1e-10, far under the 1e-6 floor).
