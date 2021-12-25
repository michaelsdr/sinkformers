import numpy as np
import matplotlib.pyplot as plt
rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)

model = 's'
iterations = [1, 3]
colors = ['#feb24c', '#99d8c9', 'grey',  'darkgreen', 'orange', 'black', 'purple']
p = 0.55
f, ax = plt.subplots(1, 4, figsize=(13 * p, 3.7 * p))

alpha = 3
lw_main = 2
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
    print(n_it, np.max(avg_test_accuracy))
    avg_train_loss = np.median(train_loss, 0)
    quantile_test_loss_1 = np.quantile(test_loss, 0.25, axis=0)
    quantile_test_accuracy_1 = np.quantile(test_accuracy, 0.25, 0)
    quantile_train_loss_1 = np.quantile(train_loss, 0.25, 0)
    quantile_train_accuracy_1 = np.quantile(train_accuracy, 0.25, 0)
    quantile_test_loss_2 = np.quantile(test_loss, 0.75, 0)
    quantile_test_accuracy_2 = np.quantile(test_accuracy, 0.75, 0)
    quantile_train_loss_2 = np.quantile(train_loss, 0.75, 0)
    quantile_train_accuracy_2 = np.quantile(train_accuracy, 0.75, 0)
    lab = 'Transformer' if n_it==1 else 'Sinkformer'
    ax[1].plot(avg_test_loss, color=color, alpha=alpha, linewidth=lw_main, label=lab)
    ax[0].plot(avg_train_loss, color=color, alpha=alpha, linewidth=0.5 * lw_main, label=lab)
    ax[2].plot(avg_train_accuracy, color=color, alpha=alpha, linewidth=lw_main, label=lab)
    ax[3].plot(avg_test_accuracy, color=color, alpha=alpha, linewidth=lw_main, label=lab)
    x = np.arange(0, avg_test_loss.shape[0])
    ax[2].fill_between(x, quantile_train_accuracy_1, quantile_train_accuracy_2, color=color, alpha=alpha/12)
    ax[3].fill_between(x, quantile_test_accuracy_1, quantile_test_accuracy_2, color=color, alpha=alpha/12)

ax[2].set_yscale('log')
ax[3].set_yscale('log')
ax[0].grid(which='major')
ax[1].grid(which='major')
ax[2].grid(which='major')
ax[3].grid(which='major')
ax[0].set_xlabel('Epoch')
ax[1].set_xlabel('Epoch')
ax[2].set_xlabel('Epoch')
ax[3].set_xlabel('Epoch')
ax[2].set_yticks([70, 80, 90, 98])
ax[2].set_yticklabels(['70 \%', '80 \%', '90 \%', '98 \%'])
ax[3].set_yticks([78, 80, 82, 84, 86])
ax[3].set_yticklabels(['78 \%', '80 \%', '82 \%', '84 \%', '86 \%'])
x_ = ax[1].set_xlabel('Epoch')
y_ = ax[1].set_ylabel('Test loss')
z_ = ax[2].set_ylabel('Train Acc')
ax[3].set_ylabel('Test Acc')
ax[0].set_ylabel('Train Loss')
ax[1].set_ylabel('Test loss')
ax[1].minorticks_off()
f.tight_layout(pad=0.1)
leg = ax[0].legend(handlelength=0.5, handletextpad=1, loc='best', borderpad=0.2)

for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.savefig('figures/' + save_dir + '.pdf', bbox_extra_artists=[x_, y_], bbox_inches='tight')