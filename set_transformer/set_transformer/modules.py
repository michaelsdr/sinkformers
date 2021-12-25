import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from set_transformer.layers import SinkhornDistance  # , SinkhornKeops
import time
import numpy as np
eps = 1

def distmat2(X, Y, div=1):
  X_sq = (X ** 2).sum(axis=-1)
  Y_sq = (Y ** 2).sum(axis=-1)
  cross_term = X.matmul(Y.transpose(1, 2))
  return (X_sq[:, :, None] + Y_sq[:, None, :] - 2 * cross_term) / (div ** 2)

def dotmat(X, Y, div=1):
  return  - X.bmm(Y.transpose(1, 2)) / div


sinkhornkeops = SinkhornDistance(eps=eps, max_iter=1, cost=dotmat)

class MABSINK(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, sinkhorn=sinkhornkeops):
        super(MABSINK, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.sinkhorn = sinkhorn

    def forward(self, Q, K):
        dim_split = self.dim_V // self.num_heads
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        sqrtV = math.sqrt(math.sqrt(self.dim_V))

        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        pi, C, U, V = self.sinkhorn(Q_ / sqrtV, K_ / sqrtV)
        A = pi
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MABSINK_Simplified(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, sinkhorn=sinkhornkeops):
        super(MABSINK_Simplified, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.sinkhorn = sinkhorn

    def forward(self, Q, K):
        dim_split = self.dim_V // self.num_heads
        Q = self.fc_q(Q)
        sqrtV = math.sqrt(math.sqrt(self.dim_V))
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = Q_
        V_ = Q_
        pi, C, U, V = self.sinkhorn(Q_ / sqrtV, K_ / sqrtV)
        n = Q_.shape[1]
        p = K_.shape[1]
        A = pi * n
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O, A

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        O, A = self.mab(X, X)
        return O, A

class SABSINK(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, sinkhorn=sinkhornkeops):
        super(SABSINK, self).__init__()
        self.mab = MABSINK_Simplified(dim_in, dim_in, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False,
                 save_attn_0=None, save_attn_1=None):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)


    def forward(self, X):
        Y = self.I.repeat(X.size(0), 1, 1)
        H_0, A_0 = self.mab0(Y, X)
        H_1, A_1 = self.mab1(X, H_0)
        return H_1, A_0, A_1

class ISABSINK(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, sinkhorn=sinkhornkeops):
        super(ISABSINK, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MABSINK(dim_out, dim_in, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)
        self.mab1 = MABSINK(dim_in, dim_out, dim_out, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):

        Y = self.I.repeat(X.size(0), 1, 1)
        H = self.mab0(Y, X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        Z = self.S.repeat(X.size(0), 1, 1)
        pma, _ = self.mab(Z, X)
        return pma

class PMASINK(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, sinkhorn=sinkhornkeops):
        super(PMASINK, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MABSINK(dim, dim, dim, num_heads, ln=ln, sinkhorn=sinkhorn)

    def forward(self, X):
        Z = self.S.repeat(X.size(0), 1, 1)
        return self.mab(Z, X)
