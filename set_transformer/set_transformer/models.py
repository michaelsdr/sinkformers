from set_transformer.modules import *
import torch.nn as nn

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads,  ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

class SetTransformerLegacy(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformerLegacy, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class ModelNet(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
        save_attn_0 = 'attn_0.npy',
        save_attn_1 = 'attn_1.npy',
    ):
        super(ModelNet, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, save_attn_0=None, save_attn_1=None),
            ISAB(dim_hidden, dim_hidden, num_heads,num_inds, ln=ln, save_attn_0=save_attn_0, save_attn_1=save_attn_1),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        Y = self.enc(X)
        return self.dec(Y).squeeze()

class ModelNetSink(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=True,
        n_it=1
    ):
        super(ModelNetSink, self).__init__()
        sinkhornkeops = SinkhornDistance(eps=eps, max_iter=n_it, cost=dotmat)
        self.enc = nn.Sequential(
            ISABSINK(dim_input, dim_hidden, num_heads, num_inds, ln=ln, sinkhorn=sinkhornkeops),
            ISABSINK(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, sinkhorn=sinkhornkeops),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMASINK(dim_hidden, num_heads, num_outputs, ln=ln, sinkhorn=sinkhornkeops),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        Y = self.enc(X)
        return self.dec(Y).squeeze()

class ModelNetSabSink(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
        n_it=1,
    ):
        super(ModelNetSabSink, self).__init__()
        sinkhornkeops = SinkhornDistance(eps=eps, max_iter=n_it, cost=distmat2)
        self.enc = nn.Sequential(
            SABSINK(dim_input, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
            SABSINK(dim_hidden, dim_hidden, num_heads, ln=ln, sinkhorn=sinkhornkeops),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMASINK(dim_hidden, num_heads, num_outputs, ln=ln, sinkhorn=sinkhornkeops),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()
