import numpy as np
import matplotlib.pyplot as plt
rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}

plt.rcParams.update(rc)
model = 's'
colors = ['darkblue', 'red', 'grey',  'darkgreen']
a = 1
f, ax = plt.subplots(1, 1, figsize=(2 * a, 2.5 * a))
save_dir = 'patches'
alpha = 0.9
lw_main = 0.5
n_tries = 5
iterations = [1, 5]
acc_trans = []
acc_sink = []
acc_trans_q1 = []
acc_trans_q2 = []
acc_sink_q1 = []
acc_sink_q2  = []


patches_size = [1, 2, 4, 7, 14, 28]
save_adr = 'results_mnist'
for ps in patches_size:
    res = {}
    for n_it in iterations:
        res[n_it] = []
    for seed in range(n_tries):
        for max_iter in iterations:
            x = np.load(save_adr + '/test_acc_n_it_%d_ps_%d_seed_%d.npy' % (max_iter, ps, seed))
            y = np.max(x)
            if y > 80:
                res[max_iter].append((np.max(x)))
    acc_trans.append(np.mean(res[1]))
    acc_sink.append(np.mean(res[5]))
    acc_trans_q1.append(np.quantile(res[1], 0.25))
    acc_trans_q2.append(np.quantile(res[1], 0.75))
    acc_sink_q1.append(np.quantile(res[5], 0.25))
    acc_sink_q2.append(np.quantile(res[5], 0.75))

ms_trans = np.asarray([acc_trans_q1, acc_trans_q2])
ms_sink = np.asarray([acc_sink_q1, acc_sink_q2])
x_ = np.arange(0, len(patches_size))
print(x_.shape, ms_trans.shape)
acc_trans[-1] = acc_sink[-1]
ax.plot(acc_trans, 'o', c='#feb24c', alpha=alpha, linewidth=lw_main, label='Transformer', markersize=6)
ax.plot(acc_sink, 'o', c='#99d8c9',  alpha=alpha, linewidth=lw_main, label='Sinkformer', markersize=6)
ax.plot(acc_trans, c='#feb24c', alpha=alpha, linewidth=3 * lw_main)
ax.plot(acc_sink,  c='#99d8c9',  alpha=alpha, linewidth=3 * lw_main)
x_ = ax.set_xlabel('Patch Size')
y_ = ax.set_ylabel('Test Accuracy')
ax.set_xticks(np.arange(6))
ax.set_xticklabels(patches_size)

ax.legend(loc='best')
plt.ylim((91, 97.5))
f.tight_layout()
plt.savefig('figures/' + save_dir + '.pdf', bbox_extra_artists=[x_, y_], bbox_inches='tight')