from set_transformer.data_modelnet40 import ModelFetcher
from set_transformer.modules import ISAB, PMA, SAB, ISABSINK
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pykeops
from set_transformer.models import  ModelNet, ModelNetSink
pykeops.set_bin_folder("/gpfswork/rech/ynt/uxe47ws/cache_keops")

#pykeops.clean_pykeops()

parser = argparse.ArgumentParser()
parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=2000)
args = parser.parse_args()
args.exp_name = f"N{args.num_pts}_d{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "result/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(log_dir)

generator = ModelFetcher(
    "../dataset/ModelNet40_cloud.h5",
    args.batch_size,
    down_sample=int(10000 / args.num_pts),
    do_standardize=True,
    do_augmentation=(args.num_pts == 5000),
)

model = ModelNetSink(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc, n_it=10)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
model = nn.DataParallel(model)
model = model.cuda()

for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for imgs, _, lbls in generator.train_data():
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
    writer.add_scalar("train_loss", avg_loss)
    writer.add_scalar("train_acc", avg_acc)
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    if epoch % 10 == 0:
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
        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("test_loss", avg_loss)
        writer.add_scalar("test_acc", avg_acc)
        print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
