from set_transformer.layers import SinkhornDistance, SinkhornKeops
import time

import torch
from torch.autograd import grad
import numpy as np

from pykeops.torch import Genred
import pykeops
from geomloss import SamplesLoss

pykeops.set_bin_folder("/gpfswork/rech/ynt/uxe47ws/cache_keops")
pykeops.clean_pykeops()

s = SamplesLoss(blur=1, n_iters=100)


B = 256
N_min = 10
N_max = 11
k = 4
num_heads = 4
D = 128
eps = 1

def distmat2(X, Y, div=1):
  X_sq = (X ** 2).sum(axis=-1)
  Y_sq = (Y ** 2).sum(axis=-1)
  cross_term = X.matmul(Y.transpose(1, 2))
  return (X_sq[:, :, None] + Y_sq[:, None, :] - 2 * cross_term) / (div ** 2)

def dotmat(X, Y, div=1):
  return X.bmm(Y.transpose(1, 2)) / div

sinkhorn = SinkhornDistance(eps, max_iter=1, cost=dotmat)
sinkhornkeops = SinkhornKeops(eps, max_iter=1, cost=dotmat)

N = np.random.randint(N_min, N_max)


X = torch.rand(B, N, D).cuda(device=1)
X.requires_grad = True

t = time.time()
pi1, C1, U1, V1 = sinkhorn(X, X)
print(time.time() - t)

t = time.time()
pi, C, U, V = sinkhornkeops(X, X)
print(time.time() - t)

print(((pi - pi1)**2).sum())


l = pi.sum()

l.backward()