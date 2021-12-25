# You need to install the following python packages
# pytorch, vit_pytorch.
import torch
import torchvision
from vit_pytorch import ViT, ViT_only_Att
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


max_iter = 1

Dpath = 'mnist'
Bs_Train = 10
Bs_Test = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nsamples = 2000
N_EPOCHS = 30

heads = 1
mlp_dim = 128

eps = 1
lr = 0.002
depth = 1
print('Examines ',  Bs_Train * nsamples, 'samples per epochs')

tform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))])

tr_set = torchvision.datasets.MNIST(Dpath, train=True, download=False,
                                    transform=tform_mnist)

tr_load = torch.utils.data.DataLoader(tr_set, batch_size=Bs_Train, shuffle=False)

ts_set = torchvision.datasets.MNIST(Dpath, train=False, download=False, transform=tform_mnist)

ts_load = torch.utils.data.DataLoader(ts_set, batch_size=Bs_Test, shuffle=True)


def train_iter(model, optimz, data_load, loss_val, save_adr):
    samples = len(data_load.dataset)
    model.train()

    for i, (data, target) in enumerate(data_load):
        if i >= nsamples:
            break
        data = data.to(device)
        target = target.to(device)
        #print(target)
        optimz.zero_grad()
        output, attn_weights = model(data)
        if i == 0:
            np.save(save_adr, np.asarray(attn_weights))
        out = F.log_softmax(output, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()

        if i % 100 == 0:
            # print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
            #       ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
            #       '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())


def evaluate(model, data_load, loss_val):
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

    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')



start_time = time.time()

model = ViT_only_Att(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=128, depth=depth, heads=heads, mlp_dim=mlp_dim, max_iter=max_iter, eps=eps).to(device)

print(depth, max_iter, eps)

#model = torch.nn.DataParallel(model).cuda()


optimz = optim.Adam(model.parameters(), lr=lr)

trloss_val, tsloss_val = [], []
for epoch in range(1, N_EPOCHS + 1):
    # if epoch == 20:
    #     for g in optimz.param_groups:
    #         print('lr /= 10')
    #         g['lr'] /= 10
    if epoch == 25:
        for g in optimz.param_groups:
            print('lr /= 10')
            g['lr'] /= 10
    print('Epoch:', epoch)
    save_adr = 'attn_weights_ps_7_sinkhorn/attn_%d.npy' % epoch
    train_iter(model, optimz, tr_load, trloss_val, save_adr)
    evaluate(model, ts_load, tsloss_val)


print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

#
# for epoch in range(41, 50):
#     if epoch == 41:
#         for g in optimz.param_groups:
#             print('lr /= 10')
#
#             g['lr'] /= 10
#     print('Epoch:', epoch)
#     save_adr = 'attn_weights_ps_7_sinkhorn/attn_%d.npy' % epoch
#     train_iter(model, optimz, tr_load, trloss_val, save_adr)
#     evaluate(model, ts_load, tsloss_val)
# #
# # for g in optimz.param_groups:
# #     g['lr'] *= 10