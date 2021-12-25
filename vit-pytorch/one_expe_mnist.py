# You need to install the following python packages
# pytorch, vit_pytorch.
import torch
import torchvision
from vit_pytorch import ViT_only_Att
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_it", type=int, default='5', help='number of iterations within sinkkorn')
parser.add_argument("--seed", type=int, default='0')
parser.add_argument("--lr", type=float, default='0.0005')
parser.add_argument("--ps", type=int, default='7', help='patch size')
args = parser.parse_args()


n_it = args.n_it
seed = args.seed
lr = args.lr
ps = args.ps


Dpath = 'mnist'
Bs_Train = 100
Bs_Test = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))])

tr_set = torchvision.datasets.MNIST(Dpath, train=True, download=True,
                                    transform=tform_mnist)

tr_load = torch.utils.data.DataLoader(tr_set, batch_size=Bs_Train, shuffle=True)

ts_set = torchvision.datasets.MNIST(Dpath, train=False, download=True, transform=tform_mnist)

ts_load = torch.utils.data.DataLoader(ts_set, batch_size=Bs_Test, shuffle=True)


def train_iter(model, optimz, data_load, loss_val, save_adr):
    samples = len(data_load.dataset)
    model.train()

    for i, (data, target) in enumerate(data_load):
        data = data.to(device)
        target = target.to(device)
        optimz.zero_grad()
        output, attn_weights = model(data)
        out = F.log_softmax(output, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()

        if i % 100 == 0:
            loss_val.append(loss.item())


def evaluate(model, data_load, loss_val, test_acc):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data)[0], dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)\n')
    test_acc.append(acc.detach().cpu().item())


def main(N_EPOCHS=45, heads=1, mlp_dim=128, max_iter=n_it, eps=1, lr=lr, depth=1,
         ps=ps, seed=seed, save_adr='results_mnist'):

    model = ViT_only_Att(image_size=28, patch_size=ps, num_classes=10, channels=1,
                dim=128, depth=depth, heads=heads, mlp_dim=mlp_dim, max_iter=max_iter, eps=eps).to(device)


    optimz = optim.Adam(model.parameters(), lr=lr)

    trloss_val, tsloss_val, test_acc = [], [], []
    for epoch in range(1, N_EPOCHS + 1):
        if epoch == 35:
            for g in optimz.param_groups:
                print('lr /= 10')
                g['lr'] /= 10
        if epoch == 41:
            for g in optimz.param_groups:
                print('lr /= 10')
                g['lr'] /= 10
        print('Epoch:', epoch)
        train_iter(model, optimz, tr_load, trloss_val, save_adr)
        evaluate(model, ts_load, tsloss_val, test_acc)
        np.save(save_adr + '/test_acc_n_it_%d_ps_%d_seed_%d' %(max_iter, ps, seed), np.asarray(test_acc))
    return test_acc


    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


res = main(N_EPOCHS=45, heads=1, mlp_dim=128, max_iter=n_it, eps=1, lr=lr, depth=1,
         ps=ps, seed=seed, save_adr='results_mnist')

