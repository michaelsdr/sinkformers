from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from vit_pytorch.vit import  ViT

def main(n_it=1, save_adr = 'sinkhorn.npy', save_model = 'sinkhorn.pth', lr=3e-5, epochs = 300, seed=0):
# Training settings
    batch_size = 64
    gamma = 0.7
    seed = seed
    resume = os.path.exists(save_model)
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


    os.makedirs('examples/data', exist_ok=True)

    train_dir = 'examples/data/train'
    test_dir = 'examples/data/test1'

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    random_idx = np.random.randint(1, len(train_list), size=9)


    train_list, valid_list = train_test_split(train_list,
                                              test_size=0.2,
                                              stratify=labels)


    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    class CatsDogsDataset(Dataset):
        def __init__(self, file_list, transform=None):
            self.file_list = file_list
            self.transform = transform

        def __len__(self):
            self.filelength = len(self.file_list)
            return self.filelength

        def __getitem__(self, idx):
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            img_transformed = self.transform(img)

            label = img_path.split("/")[-1].split(".")[0]
            label = 1 if label == "dog" else 0

            return img_transformed, label

    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
    test_data = CatsDogsDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

    device = 'cuda'

    model = ViT(image_size=224,
        patch_size=16,
        num_classes=2,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=128,
        pool = 'cls', channels = 3,
        dim_head = 64, dropout = 0.,
        emb_dropout = 0., max_iter=n_it,
        ).to(device)

    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    if resume:
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(save_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        ep = checkpoint['epoch']
        for g in optimizer.param_groups:
            if ep > 250 :
                g['lr'] = 3e-6
        losses = checkpoint['losses']
        train_loss_array, val_loss_array, train_accuracy_array,\
        val_accuracy_array = losses[0].tolist(), losses[1].tolist(), losses[2].tolist(), losses[3].tolist()
    else:
        ep = 0
        val_loss_array = []
        train_loss_array = []
        val_accuracy_array = []
        train_accuracy_array = []
    for epoch in range(ep, epochs):
        print('epoch ', epoch)
        if epoch == 250 :
            for g in optimizer.param_groups:
                g['lr'] /= 10
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output, attn_weights = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output, val_attn_weights = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        #scheduler.step()
        val_accuracy_array.append(epoch_val_accuracy.cpu().detach().numpy())
        train_accuracy_array.append(epoch_accuracy.cpu().detach().numpy())
        val_loss_array.append(epoch_val_loss.cpu().detach().numpy())
        train_loss_array.append(epoch_loss.cpu().detach().numpy())
        losses = np.asarray([train_loss_array, val_loss_array, train_accuracy_array, val_accuracy_array])
        np.save(save_adr, losses)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
        }, save_model)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
