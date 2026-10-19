"""
Training loop. SGD + momentum, patience-based LR decay -- same scheme
07_alexnet/08_vggnet/09_resnet all use (established precedent for this
repo rather than the paper's own iteration-count schedule, which assumes
a training run far longer than this repo's compute budget targets).

Momentum: paper specifies 0.99, but that's deliberately tuned for batch
size 1 (Section 3: "we favor large input tiles over a large batch size
and hence reduce the batch to a single image. Accordingly we use a high
momentum (0.99) such that a large number of the previously seen training
samples determine the update") -- i.e. 0.99 exists to substitute for
cross-example averaging a bigger batch would otherwise provide. This
repo's timing probe showed batch=8 costs under 0.2s/step (negligible),
so it isn't batch-size-constrained the way the paper's GPU-memory-bound
single-image-tile setup is, and uses momentum 0.9 (every CNN paper
in this repo's own precedent) with batch=8 instead of reproducing a
momentum value whose entire purpose doesn't apply here.

Weight maps (Eqn.2) are computed fresh per batch from each patch's own
ground-truth mask -- not precomputed once per image, since patches (and
their elastic deformation) are randomized per batch.

LR0=0.01, not 0.1: a real, confirmed instability, not a guess carried
over from another paper. A first smoke test at lr=0.1 oscillated wildly
epoch to epoch (val pixel accuracy swinging 0.79 -> 0.18 -> 0.81 -> 0.21,
i.e. flipping between predicting almost-all-foreground and
almost-all-background) and the patience-based decay didn't fix it --
even after two decays down to lr=0.001 the same flip-flop continued
(JOURNAL.md entry 6 has the full log). Dropping straight to lr=0.01 (no
decay, 10 fixed-LR epochs) gave a genuinely different, mostly-improving
pattern instead (val_acc mostly 0.72-0.87, train_loss trending down with
normal reduced-subset/small-val-set noise, same noisy-but-real character
09_resnet's JOURNAL documents for its own reduced-subset run). The paper
itself gives no numeric base LR to start from (Section 3 only specifies
momentum=0.99 and batch=1), so there was nothing to inherit here the way
09_resnet inherited the ResNet paper's own stated lr=0.1 -- this had to
be found the same way 08_vggnet found its own stable LR, by hitting the
instability first.
"""
import time
import json
import numpy as np
from pathlib import Path

from unet import UNet
from loss import WeightedPixelSoftmaxCE
from data import make_samplers
from weight_map import compute_weight_map

BATCH_SIZE = 8
STEPS_PER_EPOCH = 50
N_EPOCHS = 30
LR0 = 0.01
MOMENTUM = 0.9
PATIENCE = 3
LR_DECAY = 0.1


def batch_weights(lbls):
    return np.stack([compute_weight_map(l.astype(bool)) for l in lbls])


def pixel_accuracy(logits, labels):
    preds = logits.argmax(axis=1)
    return (preds == labels).mean()


def evaluate(model, loss_fn, val_sampler, n_batches=4):
    losses, accs = [], []
    for _ in range(n_batches):
        imgs, lbls = val_sampler.sample_batch(BATCH_SIZE, augment=False)
        weights = batch_weights(lbls)
        logits = model.forward(imgs)
        l = loss_fn.forward(logits, lbls, weights)
        losses.append(l)
        accs.append(pixel_accuracy(logits, lbls))
    return float(np.mean(losses)), float(np.mean(accs))


def train():
    train_sampler, val_sampler = make_samplers(seed=0)
    model = UNet(seed=1)
    loss_fn = WeightedPixelSoftmaxCE()
    params = model.params()
    velocity = {k: np.zeros_like(v) for k, v in params.items()}

    lr = LR0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    t_start = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        epoch_losses = []
        for step in range(STEPS_PER_EPOCH):
            imgs, lbls = train_sampler.sample_batch(BATCH_SIZE, augment=True)
            weights = batch_weights(lbls)

            logits = model.forward(imgs)
            l = loss_fn.forward(logits, lbls, weights)
            dlogits = loss_fn.backward()
            _, grads = model.backward(dlogits)

            params = model.params()
            for k in params:
                g = grads[k]
                velocity[k] = MOMENTUM * velocity[k] - lr * g
                params[k] += velocity[k]
            epoch_losses.append(l)

        val_loss, val_acc = evaluate(model, loss_fn, val_sampler)
        train_loss = float(np.mean(epoch_losses))
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)
        elapsed = time.time() - t_start
        print(f"epoch {epoch:2d}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr:.5f}  "
              f"elapsed={elapsed:.1f}s")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                lr *= LR_DECAY
                epochs_no_improve = 0
                print(f"  -> no improvement for {PATIENCE} epochs, lr decayed to {lr:.5f}")

    return model, loss_fn, train_sampler, val_sampler, history


if __name__ == "__main__":
    model, loss_fn, train_sampler, val_sampler, history = train()
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved {log_dir / 'training_log.json'}")

    from figures import plot_training_curve, plot_sample_predictions
    plot_training_curve(history)
    plot_sample_predictions(model, val_sampler)
