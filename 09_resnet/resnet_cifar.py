"""
ResNet-20 for CIFAR-10 (JOURNAL.md entry 2/3): n=3, Section 4.2's own
CIFAR-10 recipe verbatim -- conv1 (3x3,16) then 3 stages of n=3
BasicBlocks each (filter widths {16,32,64}, feature maps {32,16,8}, first
block of stages 2/3 downsamples with stride 2), global average pool,
FC-10. 6n+2 = 20 weighted layers total. Option-A identity shortcuts only
(residual_block.py) -- the paper's own choice for CIFAR-10, not a cut.
"""
import numpy as np
from conv2d import Conv2D
from bn2d import BatchNorm2d
from dense import Dense
from residual_block import BasicBlock

N_BLOCKS_PER_STAGE = 3  # n=3 -> ResNet-20 (JOURNAL.md entry 3)


class ResNet20_CIFAR:
    def __init__(self, seed=0):
        s = iter(range(seed * 1000, seed * 1000 + 100))
        self.conv1 = Conv2D(3, 16, 3, stride=1, padding=1, seed=next(s))
        self.bn1 = BatchNorm2d(16)

        self.stages = []
        self._stage_names = []
        widths = [16, 32, 64]
        in_c = 16
        for stage_idx, out_c in enumerate(widths):
            blocks = []
            for b in range(N_BLOCKS_PER_STAGE):
                stride = 2 if (stage_idx > 0 and b == 0) else 1
                blocks.append(BasicBlock(in_c, out_c, stride, seed=next(s)))
                in_c = out_c
            self.stages.append(blocks)
            self._stage_names.append([f'stage{stage_idx + 1}.block{b}' for b in range(N_BLOCKS_PER_STAGE)])

        self.fc = Dense(64, 10, activation=False, seed=next(s))

    def params(self):
        p = {
            'conv1.W': self.conv1.W, 'conv1.b': self.conv1.b,
            'bn1.gamma': self.bn1.gamma, 'bn1.beta': self.bn1.beta,
            'fc.W': self.fc.W, 'fc.b': self.fc.b,
        }
        for blocks, names in zip(self.stages, self._stage_names):
            for block, name in zip(blocks, names):
                p.update(block.params(name))
        return p

    def forward(self, x, train=True):
        out = self.conv1.forward(x)
        if train:
            out, self._bn1_cache = self.bn1.forward_train(out)
        else:
            out = self.bn1.forward_eval(out)
        self._relu1_mask = out > 0
        out = np.maximum(0, out)

        for blocks in self.stages:
            for block in blocks:
                out = block.forward(out, train)

        self._pool_shape = out.shape
        pooled = out.mean(axis=(2, 3))
        logits = self.fc.forward(pooled)
        return logits

    @staticmethod
    def softmax_ce(logits, y_onehot):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        dlogits = (probs - y_onehot) / N
        return loss, dlogits, probs

    def backward(self, dlogits):
        dpooled, dWfc, dbfc = self.fc.backward(dlogits)

        N, C, H, W = self._pool_shape
        dout = np.broadcast_to(dpooled[:, :, None, None] / (H * W), self._pool_shape).copy()

        grads = {'fc.W': dWfc, 'fc.b': dbfc}
        for blocks, names in zip(reversed(self.stages), reversed(self._stage_names)):
            for block, name in zip(reversed(blocks), reversed(names)):
                dout, block_grads = block.backward(dout, name)
                grads.update(block_grads)

        dout = dout * self._relu1_mask
        dout, dgamma1, dbeta1 = self.bn1.backward(dout, self._bn1_cache)
        dx, dW1, db1 = self.conv1.backward(dout)
        grads.update({'conv1.W': dW1, 'conv1.b': db1, 'bn1.gamma': dgamma1, 'bn1.beta': dbeta1})
        return dx, grads
