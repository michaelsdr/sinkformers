
"""Implements the (unbiased) Sinkhorn divergence between sampled measures."""

import numpy as np
import torch
from functools import partial

import pykeops

pykeops.clean_pykeops()

pykeops.set_bin_folder("/gpfswork/rech/ynt/uxe47ws/cache_keops")


try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch import generic_sum
    from pykeops.torch import Genred
except:
    keops_available = False
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
import pdb

# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================



def log_weights(α):
    α_log = α.log()
    α_log[α <= 0] = -100000
    return α_log



def sinkhorn_cost( α, β, u, v ): #compute the optimal transport cost given the samples and the potentials
    return torch.dot( α.view(-1), u.view(-1) ) + torch.dot( β.view(-1), v.view(-1) )
           

def sinkhorn_loop( softmin, marg_x, α_log, β_log, C_xy, C_yx, ε, keep_iterations = False):
#softmin : keops or tensorized (PytTorch) 
#marg_x : compute the error between the current sinkhorn 
    Nits = 1
    N, D = C_xy[0].shape
    M, D = C_xy[1].shape
    # Start with a decent initialization for the dual vectors:
    v = softmin(ε, C_yx, α_log )  # OT(α,β) wrt. a
    u = softmin(ε, C_xy, β_log )  # OT(α,β) wrt. b
    
    err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
    err_marginal = [err]
    
    if keep_iterations:
        potential_tab_u = [u - u.sum()/N]
        cost_tab = [sinkhorn_cost(α_log.exp(), β_log.exp(), u, v )]


    while (Nits< 10**3) and (err > 10**(-5) or Nits < 2 ):
        Nits += 1
        # "Coordinate ascent" on the dual problems:
        v = softmin(ε, C_yx, α_log + u/ε )  # OT(α,β) wrt. a
        u = softmin(ε, C_xy, β_log + v/ε )  # OT(α,β) wrt. b
        
        err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
        err_marginal.append(err)
        
        if keep_iterations:
            cost_tab.append( sinkhorn_cost(α_log.exp(), β_log.exp(), u, v) )
            potential_tab_u.append(u - u.sum()/N)

    print("iterations = ",Nits, "err = ", err)

    mean_u = u.sum()/N
    v, u = v + mean_u , u - mean_u

    if keep_iterations:
        return u, v, cost_tab, potential_tab_u
    else:
        return u, v
    


def sinkhorn_loop_RNA( softmin, marg_x, α_log, β_log, C_xy, C_yx, ε, Nstore = 10, keep_iterations = False ):
    Nits = 1
    N, D = C_xy[0].shape
    M, D = C_xy[1].shape
    # Start with a decent initialization for the dual vectors:
    
    v = softmin(ε, C_yx, α_log )  # OT(α,β) wrt. a
    u = softmin(ε, C_xy, β_log )  # OT(α,β) wrt. b
    mean_u = u.sum()/N
    u = u - mean_u
    v = v + mean_u
    err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
    err_marginal = [err]
    
    if keep_iterations:
        potential_tab = [u - u.sum()/N]
        cost_tab = [sinkhorn_cost(α_log.exp(), β_log.exp(), u, v )]

    store_gu = u[:,None]
    store_u = torch.zeros(store_gu.shape).type(dtype)

    while (Nits< 10**4) and ( err > 10**(-4)  ):

        u = extrapolation_RNA(store_u, store_gu)
        store_u = torch.cat( (store_u[:,1*(Nits>=Nstore):min(Nits,Nstore)], u[:,None]), 1 )
        v = softmin(ε, C_yx, α_log + u/ε )  # OT(α,β) wrt. a
        u = softmin(ε, C_xy, β_log + v/ε )  # OT(α,β) wrt. b v
        mean_u = u.sum()/N
        u = u - mean_u
        v = v + mean_u
        store_gu = torch.cat( (store_gu[:,1*(Nits>=Nstore):min(Nits,Nstore)], u[:,None]), 1 )
        err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
        err_marginal.append(err)
        
        if keep_iterations:
            potential_tab.append(u - u.sum()/N)
            cost_tab.append( sinkhorn_cost(α_log.exp(), β_log.exp(), u, v) )

        Nits += 1

            


    print("iterations = ",Nits, "err = ", err)

    if keep_iterations:
        return u, v, cost_tab, potential_tab
    else:
        return u, v

def extrapolation_RNA(xs, gxs, reg = 10**(-6)):
    d, N = xs.shape
    res = gxs - xs
    gram_res = res.T@res
    norm = (gram_res**2).sum().sqrt()
    gram_res = gram_res/norm
    weights = torch.solve( torch.ones(N,1).type(dtype), gram_res + reg*torch.eye(N).type(dtype) )[0]
    weights = weights/weights.sum()

    return (weights.T*(gxs)).sum(dim = 1)


def OT_regularized(acceleration = False, keep_iterations = False):
    
    def T(blur, alpha, x, beta, y):
        
        n, d = x.shape
        Loss =  loss_sinkhorn_online(d, p=2, blur=blur,  acceleration = acceleration, keep_iterations = keep_iterations)
        if keep_iterations:
            
            u_ab, v_ab, cost_tab_ab, potential_tab_ab_u = Loss( alpha, x, beta, y )
           
            cost_ab = torch.stack(cost_tab_ab)
            potential_ab_u = torch.stack(potential_tab_ab_u)
            
            
            return u_ab, v_ab, cost_ab, potential_ab_u
        
        else:
            u_ab, v_ab = Loss( alpha, x, beta, y )
            
            return u_ab, v_ab
        
    return T

def sinkhorn_divergence(acceleration = False, keep_iterations = False):
    
    def S(blur, alpha, x, beta, y):
        
        n, d = x.shape
        Loss =  loss_sinkhorn_online(d, p=2, blur=blur,  acceleration = acceleration, keep_iterations = keep_iterations)
        if keep_iterations:
            
            u_ab, v_ab, cost_tab_ab, potential_tab_ab_u = Loss( alpha, x, beta, y )
            u_aa, v_aa, cost_tab_aa, potential_tab_aa_u = Loss( alpha, x, alpha, x )
            u_bb, v_bb, cost_tab_bb, potential_tab_bb_u = Loss( beta, y, beta, y )
            
            cost_tab_ab = torch.stack(cost_tab_ab)
            cost_tab_aa = torch.stack(cost_tab_aa)
            cost_tab_bb = torch.stack(cost_tab_bb)
            n_ab, n_aa, n_bb = cost_tab_ab.shape[0], cost_tab_aa.shape[0], cost_tab_bb.shape[0]
            n_iter = max(n_ab, n_aa, n_bb)
        
            cost_ab, cost_aa, cost_bb = torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype)
            cost_ab[:n_ab] = cost_tab_ab
            cost_ab[n_ab:] = cost_tab_ab[-1]
            cost_aa[:n_aa] = cost_tab_aa
            cost_aa[n_aa:] = cost_tab_aa[-1]
            cost_bb[:n_bb] = cost_tab_bb
            cost_bb[n_bb:] = cost_tab_bb[-1]
            
            potential_tab_ab_u = torch.stack(potential_tab_ab_u)
            potential_tab_aa_u = torch.stack(potential_tab_aa_u)
        
            potential_ab_u, potential_aa_u = torch.zeros(n_iter, n).type(dtype), torch.zeros(n_iter, n).type(dtype)
            potential_ab_u[:n_ab,:] = potential_tab_ab_u
            potential_ab_u[n_ab:,:] = potential_tab_ab_u[-1,:]
            potential_aa_u[:n_aa,:] = potential_tab_aa_u
            potential_aa_u[n_aa:] = potential_tab_aa_u[-1,:]
            
            return u_ab - u_aa, v_ab - v_bb, cost_ab - .5*cost_aa - .5*cost_bb, potential_ab_u - potential_aa_u 
        
        else:
            u_ab, v_ab = Loss( alpha, x, beta, y )
            u_aa, v_aa = Loss( alpha, x, alpha, x )
            u_bb, v_bb = Loss( beta, y, beta, y )
            
            return u_ab - u_aa, v_ab - v_bb
        
    return S 

def sinkhorn_divergence_2(acceleration = False, keep_iterations = False):
    
    def S(blur, alpha, x, beta, y):
        
        n, d = x.shape
        Loss =  loss_sinkhorn_online(d, p=2, blur=blur,  acceleration = acceleration, keep_iterations = keep_iterations)
        if keep_iterations:
            
            u_ab, v_ab, cost_tab_ab, potential_tab_ab_u = Loss( alpha, x, beta, y )
            u_aa, v_aa, cost_tab_aa, potential_tab_aa_u = Loss( alpha, x, alpha, x )
            u_bb, v_bb, cost_tab_bb, potential_tab_bb_u = Loss( beta, y, beta, y )
            
            cost_tab_ab = torch.stack(cost_tab_ab)
            cost_tab_aa = torch.stack(cost_tab_aa)
            cost_tab_bb = torch.stack(cost_tab_bb)
            n_ab, n_aa, n_bb = cost_tab_ab.shape[0], cost_tab_aa.shape[0], cost_tab_bb.shape[0]
            n_iter = max(n_ab, n_aa, n_bb)
        
            cost_ab, cost_aa, cost_bb = torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype)
            cost_ab[:n_ab] = cost_tab_ab
            cost_ab[n_ab:] = cost_tab_ab[-1]
            cost_aa[:n_aa] = cost_tab_aa
            cost_aa[n_aa:] = cost_tab_aa[-1]
            cost_bb[:n_bb] = cost_tab_bb
            cost_bb[n_bb:] = cost_tab_bb[-1]
            
            potential_tab_ab_u = torch.stack(potential_tab_ab_u)
            potential_tab_aa_u = torch.stack(potential_tab_aa_u)
        
            potential_ab_u, potential_aa_u = torch.zeros(n_iter, n).type(dtype), torch.zeros(n_iter, n).type(dtype)
            potential_ab_u[:n_ab,:] = potential_tab_ab_u
            potential_ab_u[n_ab:,:] = potential_tab_ab_u[-1,:]
            potential_aa_u[:n_aa,:] = potential_tab_aa_u
            potential_aa_u[n_aa:] = potential_tab_aa_u[-1,:]
            
            return u_ab - u_aa, v_ab - v_bb, cost_ab - .5*cost_aa - .5*cost_bb, potential_ab_u - potential_aa_u 
        
        else:
            u_ab, v_ab = Loss( alpha, x, beta, y )
            
            n2 = int( np.floor(alpha.shape[0]/2) )
            m2 = int( np.floor(beta.shape[0]/2) )
            alpha1, alpha2 = alpha[:n2], alpha[n2:]
            x1,x2 = x[:n2,:], x[:n2,:]
            beta1, beta2 = beta[:m2], beta[m2:]
            y1,y2 = y[:m2,:], y[:m2,:]
            u_aa, v_aa = Loss( alpha1, x1, alpha2, x2 )
            u_bb, v_bb = Loss( beta1, y1, beta2, y2 )
            
            return u_ab - u_aa, v_ab - v_bb
        
    return S 

def richardson_sinkhorn_divergence1(acceleration = False, keep_iterations = False):
    
    S = sinkhorn_divergence(acceleration = acceleration, keep_iterations = keep_iterations)
    
    def R(blur, alpha, x, beta, y):
        n, d = x.shape

        if keep_iterations:
            
            u1, v1, cost1, potential1 = S(blur, alpha, x, beta, y)
            u2, v2, cost2, potential2 = S(np.sqrt(2)*blur, alpha, x, beta, y )
            
            n1, n2 = cost1.shape[0], cost2.shape[0]
            n_iter = max(n1, n2)
        
            cost_1, cost_2 = torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype)
            cost_1[:n1] = cost1
            cost_1[n1:] = cost1[-1]
            cost_2[:n2] = cost2
            cost_2[n2:] = cost2[-1]

            potential_1, potential_2 = torch.zeros(n_iter, n).type(dtype), torch.zeros(n_iter, n).type(dtype)
            potential_1[:n1,:] = potential1
            potential_1[n1:,:] = potential1[-1,:]
            potential_2[:n2,:] = potential2
            potential_2[n2:,:] = potential2[-1,:]
            
            return 2*u1 - u2, 2*v1 - v2, 2*cost_1 - cost_2, 2*potential_1 - potential_2
            
        else:
            u1, v1 = S(blur, alpha, x, beta, y)
            u2, v2 = S(np.sqrt(2)*blur, alpha, x, beta, y )
            return 2*u1 - u2, 2*v1 - v2
    return R

def richardson_sinkhorn_divergence2(acceleration = False, keep_iterations = False):
    
    S = sinkhorn_divergence_2(acceleration = acceleration, keep_iterations = keep_iterations)
    
    def R(blur, alpha, x, beta, y):
        n, d = x.shape

        if keep_iterations:
            
            u1, v1, cost1, potential1 = S(blur, alpha, x, beta, y)
            u2, v2, cost2, potential2 = S(np.sqrt(2)*blur, alpha, x, beta, y )
            
            n1, n2 = cost1.shape[0], cost2.shape[0]
            n_iter = max(n1, n2)
        
            cost_1, cost_2 = torch.zeros(n_iter).type(dtype), torch.zeros(n_iter).type(dtype)
            cost_1[:n1] = cost1
            cost_1[n1:] = cost1[-1]
            cost_2[:n2] = cost2
            cost_2[n2:] = cost2[-1]

            potential_1, potential_2 = torch.zeros(n_iter, n).type(dtype), torch.zeros(n_iter, n).type(dtype)
            potential_1[:n1,:] = potential1
            potential_1[n1:,:] = potential1[-1,:]
            potential_2[:n2,:] = potential2
            potential_2[n2:,:] = potential2[-1,:]
            
            return 2*u1 - u2, 2*v1 - v2, 2*cost_1 - cost_2, 2*potential_1 - potential_2
            
        else:
            u1, v1 = S(blur, alpha, x, beta, y)
            u2, v2 = S(np.sqrt(2)*blur, alpha, x, beta, y )
            return 2*u1 - u2, 2*v1 - v2
    return R

# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def squared_distances(x, y):
    D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
    D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
    D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    return D_xx - 2*D_xy + D_yy

def softmin_tensorized():
    def softmin(ε, C_xy, g):
        x, y = C_xy
        return - ε * ( g[None,:] - squared_distances(x,y)/ε ).logsumexp(1).view(-1)
    return softmin

def marginal_tensorized():
    print("Tensorized version")
    def marginal(ε, C_xy, u, v):
        x, y = C_xy
        marginal_i = torch.exp( (u[:,None] + v[None,:] - squared_distances(x,y))/ε ).sum(dim = 1)
        return marginal_i
    return marginal
# ==============================================================================
#                          backend == "online"
# ==============================================================================

cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y))",
}

def softmin_online(ε, C_xy, f_y, log_conv=None):
    x, y = C_xy
    # KeOps is pretty picky on the input shapes...
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)

def marginal_online(ε, C_xy, b_x, a_y, log_conv=None):
    x,y = C_xy
    
    return log_conv( torch.Tensor([1/ε]).type_as(x), x, y, b_x.view(-1,1), a_y.view(-1,1) )



def keops_OT_plan(D, dtype="float32"):
        
    OT_plan = Genred('Exp( (F_i + G_j - SqDist(X_i,Y_j)) * E )', # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                       ['E = Pm(1)',          # First arg  is a parameter,    of dim 1
                        'X_i = Vi({})'.format(D),          # Second arg is indexed by "i", of dim 3
                        'Y_j = Vj({})'.format(D),          # Third arg  is indexed by "j", of dim 3
                        'F_i = Vi(1)',  # 4th arg: one scalar value per line
                        'G_j = Vj(1)'],         # Fourth arg is indexed by "j", of dim 2
                       reduction_op='Sum',
                       axis=1)                # Summation over "j"

    return OT_plan

def keops_lse(cost, D, dtype="float32"):
    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )",
                                 "A = Vi(1)",
                                 "X = Vi({})".format(D),
                                 "Y = Vj({})".format(D),
                                 "B = Vj(1)",
                                 "P = Pm(1)",
                                 dtype = dtype)
    return log_conv

def extrapolate_potential(g, C, log_conv):
    x_, y = C
    def f_potential(x):
        return - ε * log_conv( x, y, g.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)
    return f_potential

def extrapolate_potential_debiaised(g_y, g_x, C_x, C_y, log_conv):
    x_, y = C_y
    x_, xx = C_x
    def f_potential(x):
        return - ε * log_conv( x, y, g_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1) +  ε * log_conv( x, xx, g_x.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)


def loss_sinkhorn_online(dim, p=2, blur =.05, acceleration = False, keep_iterations = False, extrapolate = False, keops_available = True):
    
    cost = cost_formulas[p]
    
    if keops_available:
        
        softmin = partial( softmin_online, log_conv = keops_lse(cost, dim, dtype="float32") ) 
        marg_x = partial( marginal_online, log_conv = keops_OT_plan(dim, dtype="float32" ) )
    
    else:
        softmin = softmin_tensorized()
        marg_x = marginal_tensorized()
        
    
    if acceleration:
        def loss(α, x, β, y):
            # The "cost matrices" are implicitely encoded in the point clouds,
            # and re-computed on-the-fly:
            C_xy, C_yx = ( (x, y.detach()), (y, x.detach()) )
            if keep_iterations:
                u, v, cost_tab, potential_tab = sinkhorn_loop_RNA( softmin, marg_x,
                                                    log_weights(α), log_weights(β), 
                                                   C_xy, C_yx, blur, keep_iterations = keep_iterations)
        
                return u,v, cost_tab, potential_tab
            else:
                u, v = sinkhorn_loop_RNA( softmin, marg_x,
                                                    log_weights(α), log_weights(β), 
                                                   C_xy, C_yx, blur)
                return u,v

    else:
        
        def loss(α, x, β, y):
            # The "cost matrices" are implicitely encoded in the point clouds,
            # and re-computed on-the-fly:
            C_xy, C_yx = ( (x, y.detach()), (y, x.detach()) )
            if keep_iterations:
                u, v, cost_tab, potential_tab_u = sinkhorn_loop( softmin, marg_x,
                                                    log_weights(α), log_weights(β), 
                                                    C_xy, C_yx, blur, keep_iterations = keep_iterations)
                return u,v, cost_tab, potential_tab_u

            else:
                u, v = sinkhorn_loop( softmin, marg_x,
                                                    log_weights(α), log_weights(β), 
                                                    C_xy, C_yx, blur)
                return u,v
        
    return loss