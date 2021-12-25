import numpy as np
import matplotlib.pyplot as plt
rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)

fig = plt.figure(figsize=(4.5 * 0.53, 5 * 0.57))
n = 30
colors = plt.cm.rainbow(np.linspace(0, 1, n))
i = 2
attn_weights = np.load('attn_weights_ps_7/attn_%d.npy' % i)
attn = attn_weights
plt.semilogy(np.sort(np.sum(attn, 3).ravel()), label=str(i-1), c = colors[i], lw=2)
for i in range(1,  24, 4):
    if (i == 1):
        pass
    else:
        attn_weights = np.load('attn_weights_ps_7/attn_%d.npy' % i)
        attn = attn_weights
        plt.semilogy(np.sort(np.sum(attn, 3).ravel()), label=str(i-1), c = colors[i], lw=2)
plt.ylim(0.05, 20)
plt.xlabel('Sorted columns by \n sum of coefficients')
plt.ylabel('Sum of coefficients')
plt.legend(loc='best', ncol=2, fontsize=9)
fig.tight_layout()
plt.savefig('figures/softmax_learning.pdf')
#plt.show()