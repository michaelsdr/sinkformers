import numpy as np
import matplotlib.pyplot as plt
rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)

model = 'sinkformer'
train_epochs = 200
learning_rate = 1e-3
#seed = 0
p = 0.54
num_pts = 1000
iterations = [1, 9]  # Number of iterations in Sinkhorn's algorithm used when training.
colors = ['#feb24c', '#3182bd', '#99d8c9', 'grey',  'darkgreen', 'orange', 'black', 'purple']
f, ax = plt.subplots(2, 2, figsize=(7 * p, 5.1 * p))

alpha = 4
lw_main = 1
n_tries = 1

res = {}
for n_it in iterations:
    res[n_it] = []

save_dir = 'results'

for n_it in iterations:
    for seed in range(n_tries):
        save_adr = save_dir + '/%s_epochs_%d_nit_%d_lr_%f_np_%d_s_%d.npy' % (model, train_epochs, n_it, learning_rate, num_pts, seed)
        x = np.load(save_adr)
        res[n_it].append(x)


for n_it, color in zip(iterations, colors):

    X = np.asarray(res[n_it])
    x = np.median(X, 0)
    avg_losss, avg_accs, avg_test_losss, avg_test_accs = x.T[:, 0], x.T[:, 1], x.T[:, 2], x.T[:, 3]
    lab = label=str(n_it) + ' it.'
    ax[0, 0].plot(100 * (1 - avg_test_accs), color=color, alpha=alpha, linewidth=lw_main, label=lab)
    ax[1, 0].plot(100 * (1 - avg_accs), color=color, alpha=alpha, linewidth=lw_main, label=lab)
    ax[0, 1].plot(avg_test_losss, color=color, alpha=alpha, linewidth=lw_main, label=lab)
    ax[1, 1].plot(avg_losss, color=color, alpha=alpha, linewidth=lw_main, label=lab)
    avg_losss, avg_accs, avg_test_losss, avg_test_accs = X[:, 0, :], X[:, 1, :], X[:, 2, :], X[:, 3, :]
    quantile_test_loss_1 = np.quantile(avg_test_losss, 0.25, axis=0)
    quantile_test_accuracy_1 = np.quantile(100 * (1 - avg_test_accs), 0.25, 0)
    quantile_train_loss_1 = np.quantile(avg_losss, 0.25, 0)
    quantile_train_accuracy_1 = np.quantile(100 * (1 - avg_accs), 0.25, 0)
    quantile_test_loss_2 = np.quantile(avg_test_losss, 0.75, axis=0)
    quantile_test_accuracy_2 = np.quantile(100 * (1 - avg_test_accs), 0.75, 0)
    quantile_train_loss_2 = np.quantile(avg_losss, 0.75, 0)
    quantile_train_accuracy_2 = np.quantile(100 * (1 - avg_accs), 0.75, 0)
    x_ = np.arange(x.shape[1])
    ax[0, 1].fill_between(x_, quantile_test_loss_1, quantile_test_loss_2, color=color, alpha=alpha / 6)
    ax[1, 1].fill_between(x_, quantile_train_loss_1, quantile_train_loss_2, color=color, alpha=alpha / 6)
    ax[1, 0].fill_between(x_, quantile_train_accuracy_1, quantile_train_accuracy_2, color=color, alpha=alpha / 6)
    ax[0, 0].fill_between(x_, quantile_test_accuracy_1, quantile_test_accuracy_2, color=color, alpha=alpha / 6)

ax[0, 0].set_yscale('log')
ax[1, 0].set_yscale('log')
ax[0, 1].set_yscale('log')
ax[1, 1].set_yscale('log')
ax[0, 0].grid(which='major')
ax[1, 0].grid(which='major')
ax[0, 1].grid(which='major')
ax[1, 1].grid(which='major')
ax[0, 0].set_xlabel('Epoch')
ax[1, 0].set_xlabel('Epoch')
ax[0, 1].set_xlabel('Epoch')
ax[0, 0].set_yticks([10, 15, 20, 30, 40, 60])
ax[0, 0].set_yticklabels(['10 \%', '15 \%', '20 \%', '30 \%', '40 \%', '60 \%'])
ax[1, 0].set_yticks([1, 5, 10])
ax[1, 1].set_yticks([1])
ax[1, 1].set_yticklabels([1])
ax[0, 1].set_yticks([1])
ax[0, 1].set_yticklabels([1])
ax[1, 0].set_yticklabels(['1 \%', '5 \%', '10 \%'])
x_ = ax[1, 1].set_xlabel('Epoch')
y_ = ax[0, 0].set_ylabel('Test error')
ax[1, 0].set_ylabel('Train error')
ax[0, 1].set_ylabel('Test loss')
ax[1, 1].set_ylabel('Train loss')
ax[0, 1].minorticks_off()
ax[0, 1].legend(loc='upper right')
f.tight_layout(pad=0.1)
plt.savefig('figures/compare_' + save_dir + '_modelnet.pdf', bbox_extra_artists=[x_, y_], bbox_inches='tight')