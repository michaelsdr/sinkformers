#import pykeops
from set_transformer.data_modelnet40 import ModelFetcher
import numpy as np
import torch
import torch.nn as nn
from set_transformer.models import  ModelNet , ModelNetSabSink, ModelNetSink
import os

def main(model, n_it, num_pts, learning_rate, batch_size, dim, n_heads, n_anc, train_epochs, save_adr):
    generator = ModelFetcher(
        "../dataset/ModelNet40_cloud.h5",
        batch_size,
        down_sample=int(10000 / num_pts),
        do_standardize=True,
        do_augmentation=(num_pts == 5000),
    )
    if model == 'sinkformer':
        model = ModelNetSink(dim_hidden=dim, num_heads=n_heads, num_inds=n_anc, n_it=n_it)
    else:
        model = ModelNet(dim_hidden=dim, num_heads=n_heads, num_inds=n_anc)
    lr_list = np.ones(train_epochs) * learning_rate
    lr_list[3 * train_epochs // 4: ] /= 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[0])
    criterion = nn.CrossEntropyLoss()
    model = nn.DataParallel(model)
    model = model.cuda()
    avg_losss, avg_accs, avg_test_losss, avg_test_accs = [], [], [], []
    for epoch in range(train_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[epoch]
        model.train()
        losses, total, correct = [], 0, 0
        for idx, (imgs, _, lbls) in enumerate(generator.train_data()):
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        avg_losss.append(avg_loss)
        avg_accs.append((avg_acc))
        print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")


        model.eval()
        losses, total, correct = [], 0, 0
        for imgs, _, lbls in generator.test_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()
        avg_test_loss, avg_test_acc = np.mean(losses), correct / total
        avg_test_losss.append(avg_test_loss)
        avg_test_accs.append((avg_test_acc))
        print(f"Epoch {epoch}: test loss {avg_test_loss:.3f} test acc {avg_test_acc:.3f}")
        if save_adr is not None:
            np.save(save_adr, np.array([avg_losss, avg_accs, avg_test_losss, avg_test_accs]))
        state = {
            'net': model.state_dict(),
            'acc': avg_test_acc,
            'epoch': epoch,
            'perf' : np.array([avg_losss, avg_accs, avg_test_losss, avg_test_accs])
        }
        if not os.path.isdir('checkpoint_ModelNet'):
            os.mkdir('checkpoint_ModelNet')
        torch.save(state, './checkpoint_ModelNet/%s' % (save_adr[:-4] + '.pth'))
    return np.array([avg_losss, avg_accs, avg_test_losss, avg_test_accs])