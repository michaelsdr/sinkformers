import numpy as np
import matplotlib.pyplot as plt

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)
n = 10
p = 2
seed = 4
rng = np.random.RandomState(seed)
thresh = 10**(-12)
X = 0.3 * rng.randn(n//2, p)
Y = 0.3 + 0.3 * rng.rand(n//2, p)
Q = 20 * np.concatenate((X, Y), axis=0)

X = 0.3 * rng.rand(n//2, p) + np.asarray([-0.2, 0.3])
Y = 0.5 + 0.3 * rng.rand(n//2, p) + np.asarray([-0.2, -0.7])
W = 20 * np.concatenate((X, Y), axis=0)

C = np.dot(Q, W.T)
K0 = np.exp(-C)
K0 /= np.sum(K0)

K1 = K0 / np.sum(K0, axis=0)
K1 /= np.sum(K1)
K2 = K1.copy()
for i in range(100):
    K2 /= np.sum(K2, axis=0)
    K2 /= np.sum(K2, axis=1, keepdims=True)
K2 /= np.sum(K2)
l = 0.7
fig, ax = plt.subplots(1, 3, figsize=(5 * l, 2 * l))
scale = 3 * n
c = ['darkblue']
titles = ['No normalization', 'Softmax', 'Sinkhorn']

for idx, K in enumerate([K0, K1, K2]):
    for i in range(n):
        for j in range(n):
            if K[i, j] > thresh :
                if idx ==0:
                    ax[idx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.1 * scale * K[i, j], zorder=1)

                elif idx == 1:
                    ax[idx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.3 * scale * K[i, j],
                                 zorder=1)
                else:
                    ax[idx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.3 * scale * K[i, j], zorder=1)
    ax[idx].scatter(Q[:, 0], Q[:, 1], label='Queries', lw=.5, c='#dd1c77', edgecolors="k", s=20, zorder=2)
    ax[idx].scatter(W[:, 0], W[:, 1], label='Keys', lw=.5, c='#a8ddb5',  edgecolors="k", s=20, zorder=2)
    if idx in [0,1]:
        ax[idx].set_title('$K^{%d}$' %idx)
    else:
        ax[idx].set_title('$K^{\infty}$')

    ax[idx].set_xticklabels(())
    ax[idx].set_yticklabels(())

fig.tight_layout()
plt.savefig('figures/first_fig_%d.pdf' % seed)
