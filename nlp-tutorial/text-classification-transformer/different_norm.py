import torch

import matplotlib.pyplot as plt
from sinkhorn import SinkhornDistance

f, ax = plt.subplots(2, 2, figsize=(4, 4))
c = 'Pastel1'
n = 5
sink = SinkhornDistance(1, max_iter=99)
X = torch.randn((n, n)) * 4
Y = sink(X.view(1, n, n))[0]
vmin = torch.min(X)
vmax = torch.max(torch.exp(X))
ax[0, 0].imshow(X, cmap=c)
ax[0, 0].set_axis_off()
ax[0, 1].imshow(torch.exp(X), cmap=c)
ax[0, 1].set_axis_off()
ax[1, 0].imshow(X.softmax(-1), cmap=c)
ax[1, 0].set_axis_off()
ax[1, 1].imshow(Y.detach()[0], cmap=c)
ax[1, 1].set_axis_off()

plt.savefig('figures/4_attention.pdf')
