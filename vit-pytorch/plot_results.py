import numpy as np
import matplotlib.pyplot as plt
rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)
model = 's'
iterations = [1, 3]
colors = ['#feb24c', '#99d8c9', 'grey',  'darkgreen', 'orange', 'black', 'purple']
f, ax = plt.subplots(1, 1, figsize=(2 * 1.1, 2.5 * 1.1))

alpha = 3
lw_main = 1
n_tries = 1

res = {}
for n_it in iterations:
    res[n_it] = []

save_dir = 'results'

for n_it in iterations:
    for seed in range(n_tries):
        save_adr = save_dir + '/%d_it_%d.npy' % (n_it, seed)
        x = np.load(save_adr)
        res[n_it].append(x)


for n_it, color in zip(iterations, colors):

    test_loss = np.asarray(res[n_it])[:, 1, :]
    train_loss = np.asarray(res[n_it])[:, 0, :]
    train_accuracy = np.asarray(res[n_it])[:, 2, :]
    test_accuracy = np.asarray(res[n_it])[:, 3, :]
    avg_test_loss = np.median(test_loss, 0)
    avg_train_accuracy = np.median(train_accuracy, 0)
    avg_test_accuracy = np.median(test_accuracy, 0)
    avg_train_loss = np.median(train_loss, 0)
    lab = 'Transformer' if n_it == 1 else 'Sinkformer'
    x_ = np.arange(0,  x.shape[1])
    ax.plot(x_, 100 * avg_train_accuracy, '--', color=color, alpha=alpha, linewidth=lw_main)
    ax.plot(x_, 100 * avg_test_accuracy, color=color, alpha=alpha, linewidth=lw_main, label=lab)



ax.set_yscale('log')
ax.grid(which='major')
ax.set_xlabel('Epoch')
x_ = ax.set_xlabel('Epoch')
z_ = ax.set_ylabel('Accuracy')
ax.set_xticks([50, 150, 250])
ax.set_yticks([70, 75, 80, 85])
ax.set_yticklabels(['70 \%', '75 \%', '80 \%', '85 \%'])
leg = ax.legend(loc='best')

for line in leg.get_lines():
    line.set_linewidth(2.0)

f.tight_layout()
plt.savefig('figures/compare_' + save_dir + '.pdf', bbox_extra_artists=[x_, z_], bbox_inches='tight')