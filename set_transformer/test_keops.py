
!module load cuda/10.2
!module load cmake

import numpy as np
import torch

import pykeops

pykeops.clean_pykeops()

pykeops.set_bin_folder("/gpfswork/rech/ynt/uxe47ws/cache_keops")


from pykeops.torch import LazyTensor
from pykeops.numpy import LazyTensor as LazyTensor_np

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

M, N = (100000, 200000) if use_cuda else (1000, 2000)
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)


x_i = LazyTensor_np(
    x[:, None, :]
)
y_j = LazyTensor_np(
    y[None, :, :]
)
D_ij = ((x_i - y_j) ** 2).sum(-1)


D = 10
x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)
b = torch.randn(N, 4).type(tensor)

x.requires_grad = True

x_i = LazyTensor(x[:, None, :])
y_j = LazyTensor(y[None, :, :])

D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()
K_ij = (-D_ij).exp()
a_i = K_ij @ b

print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape))