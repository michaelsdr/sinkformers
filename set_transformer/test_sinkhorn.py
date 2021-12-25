import torch
import torch.nn as nn
import math
from set_transformer.layers import SinkhornDistance, SinkhornKeops

eps = 1


def distmat2(X, Y, div=1):
    X_sq = (X ** 2).sum(axis=-1)
    Y_sq = (Y ** 2).sum(axis=-1)
    cross_term = X.matmul(Y.transpose(1, 2))
    return (X_sq[:, :, None] + Y_sq[:, None, :] - 2 * cross_term) / (div ** 2)


def dotmat(X, Y, div=1):
    return - X.bmm(Y.transpose(1, 2)) / div


sinkhornkeops = SinkhornKeops(eps=eps, max_iter=1, cost=dotmat)

dim_Q, dim_K, dim_V = 256, 256, 256

dim_V = dim_V
num_heads = 4
fc_q = nn.Linear(dim_Q, dim_V)
fc_k = nn.Linear(dim_K, dim_V)
fc_v = nn.Linear(dim_K, dim_V)

fc_o = nn.Linear(dim_V, dim_V)




dim_split = dim_V // num_heads
Q = torch.rand(64, 1000, 256)
K = torch.rand(64, 16, 256)
Q, K, V = fc_q(Q), fc_k(K), fc_v(K)

sqrtV = math.sqrt(math.sqrt(dim_V))

Q_ = torch.cat(Q.split(dim_split, 2), 0)
K_ = torch.cat(K.split(dim_split, 2), 0)
V_ = torch.cat(V.split(dim_split, 2), 0)

pi, C, U, V = sinkhornkeops(Q_ / sqrtV, K_ / sqrtV)
n = Q_.shape[1]
p = K_.shape[1]
A = pi * n
cost = dotmat(Q_, K_)
B = torch.softmax(- cost / math.sqrt(dim_V), 2)
print(((A - B)**2).sum())