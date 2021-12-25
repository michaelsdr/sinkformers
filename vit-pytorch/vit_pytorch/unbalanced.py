import torch
import torch.nn as nn
import numpy as np
# keops_available = True
# try:  # Import the keops library, www.kernel-operations.io
#     from pykeops.torch import generic_logsumexp
#     from pykeops.torch import generic_sum
#     from pykeops.torch import Genred
# except:
#     keops_available = False
# from pykeops.torch import LazyTensor
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff


class SinkhornUnbalanced(nn.Module):

    def __init__(self, eps, max_iter, p=1, q=10, reduction='none'):
        super(SinkhornUnbalanced, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.p = p
        self.q = q

    def forward(self, c):
        # The Sinkhorn algorithm takes as input three variables :
          # Wasserstein cost function
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        if mu.dim() < 2:
            mu = mu.view(-1, 1)

        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Stopping criterion
        thresh = 1e-12

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = (self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u) #(self.p / (self.p + self.eps)) *
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = (self.q / (self.q + self.eps)) * (self.eps * (torch.log(nu) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v)
                # v = v.detach().requires_grad_(False)
                # v[v == float('inf')] = 0.0
                # v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                print('breaking')
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))


        return pi, C, U, V

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
