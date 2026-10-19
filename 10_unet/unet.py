"""
Composed U-Net, scoped per JOURNAL.md entry 3/4: 2 downsampling steps (3
resolution levels) instead of the paper's 4, channels 8/16/32 instead of
64/128/256/512/1024. Input 68x68x1 -> output 28x28x2 (arithmetic below,
matches JOURNAL.md entry 4 exactly).

Dimension trace (valid convs shrink by 4 per double-conv, maxpool halves,
up-conv doubles):
  in                              68  (1ch)
  level0 double-conv           -> 64  (8ch)   [skip0]
  maxpool2                     -> 32
  level1 double-conv           -> 28  (16ch)  [skip1]
  maxpool2                     -> 14
  bottleneck double-conv       -> 10  (32ch)
  up-conv (32->16ch)           -> 20
  crop skip1 (28->20) + concat -> 20  (32ch)
  level1 double-conv (dec)     -> 16  (16ch)
  up-conv (16->8ch)            -> 32
  crop skip0 (64->32) + concat -> 32  (16ch)
  level0 double-conv (dec)     -> 28  (8ch)
  1x1 conv                     -> 28  (2ch = K classes)
"""
from conv2d import Conv2D
from maxpool2d import MaxPool2x2
from transposed_conv2d import TransposedConv2x2Stride2
from crop_concat import CropConcat
from double_conv import DoubleConv3x3ReLU

C0, C1, C2 = 8, 16, 32  # level0, level1, bottleneck channel widths
K = 2  # foreground/background


class UNet:
    def __init__(self, seed=0):
        s = seed
        self.enc0 = DoubleConv3x3ReLU(1, C0, C0, seed=s + 0)
        self.pool0 = MaxPool2x2()
        self.enc1 = DoubleConv3x3ReLU(C0, C1, C1, seed=s + 10)
        self.pool1 = MaxPool2x2()
        self.bottleneck = DoubleConv3x3ReLU(C1, C2, C2, seed=s + 20)

        self.up1 = TransposedConv2x2Stride2(C2, C1, seed=s + 30)
        self.cc1 = CropConcat()
        self.dec1 = DoubleConv3x3ReLU(C1 + C1, C1, C1, seed=s + 40)

        self.up0 = TransposedConv2x2Stride2(C1, C0, seed=s + 50)
        self.cc0 = CropConcat()
        self.dec0 = DoubleConv3x3ReLU(C0 + C0, C0, C0, seed=s + 60)

        self.classifier = Conv2D(C0, K, 1, seed=s + 70)

    def forward(self, x):
        skip0 = self.enc0.forward(x)
        p0 = self.pool0.forward(skip0)
        skip1 = self.enc1.forward(p0)
        p1 = self.pool1.forward(skip1)
        bn = self.bottleneck.forward(p1)

        u1 = self.up1.forward(bn)
        cat1 = self.cc1.forward(skip1, u1)
        d1 = self.dec1.forward(cat1)

        u0 = self.up0.forward(d1)
        cat0 = self.cc0.forward(skip0, u0)
        d0 = self.dec0.forward(cat0)

        logits = self.classifier.forward(d0)
        return logits

    def backward(self, dlogits):
        grads = {}
        dd0, dW_cls, db_cls = self.classifier.backward(dlogits)
        grads['classifier.W'], grads['classifier.b'] = dW_cls, db_cls

        dcat0, dec0_grads = self.dec0.backward(dd0)
        grads.update({f'dec0.{k}': v for k, v in dec0_grads.items()})
        dskip0, du0 = self.cc0.backward(dcat0)
        dd1, dW_up0, db_up0 = self.up0.backward(du0)
        grads['up0.W'], grads['up0.b'] = dW_up0, db_up0

        dcat1, dec1_grads = self.dec1.backward(dd1)
        grads.update({f'dec1.{k}': v for k, v in dec1_grads.items()})
        dskip1_from_dec, du1 = self.cc1.backward(dcat1)
        dbn, dW_up1, db_up1 = self.up1.backward(du1)
        grads['up1.W'], grads['up1.b'] = dW_up1, db_up1

        dp1, bn_grads = self.bottleneck.backward(dbn)
        grads.update({f'bottleneck.{k}': v for k, v in bn_grads.items()})
        dp1_total = dp1  # bottleneck backward output already is d(pool1 output)

        dskip1_from_pool = self.pool1.backward(dp1_total)
        dskip1 = dskip1_from_dec + dskip1_from_pool
        dp0, enc1_grads = self.enc1.backward(dskip1)
        grads.update({f'enc1.{k}': v for k, v in enc1_grads.items()})

        dskip0_from_pool = self.pool0.backward(dp0)
        dskip0_total = dskip0 + dskip0_from_pool
        dx, enc0_grads = self.enc0.backward(dskip0_total)
        grads.update({f'enc0.{k}': v for k, v in enc0_grads.items()})
        return dx, grads

    def params(self):
        p = {'classifier.W': self.classifier.W, 'classifier.b': self.classifier.b,
             'up0.W': self.up0.W, 'up0.b': self.up0.b,
             'up1.W': self.up1.W, 'up1.b': self.up1.b}
        for name, mod in [('enc0', self.enc0), ('enc1', self.enc1),
                           ('bottleneck', self.bottleneck),
                           ('dec1', self.dec1), ('dec0', self.dec0)]:
            for k, v in mod.params().items():
                p[f'{name}.{k}'] = v
        return p
