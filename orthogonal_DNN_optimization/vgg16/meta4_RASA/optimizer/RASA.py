import torch
import torch.nn as nn
import numpy as np
from Model.Retraction import Retraction

def othogonal_projection(M, M_grad):
    A = M.T.mm(M_grad)
    A_sym = 0.5*(A + A.T)
    new = M_grad - M.mm(A_sym)
    return new

my_retraction = Retraction(1)

def retraction(M, P):
    A = M + P
    Q, R = A.qr()
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out

class RASA(nn.Module):
    def __init__(self, opt):
        super(RASA, self).__init__()
        self.beta = opt.rasa_beta
        self.decay = opt.weight_decay

    def step(self, param, e_lr, s_lr):
        # update euc
        # e_i_num = 0
        new_e_param = []
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            e_i_grad = e_i.grad
            e_i_grad = e_i_grad.add(e_i.data, alpha=self.decay)
            # e_i_before = e_i
            # if torch.equal(e_i.grad, torch.zeros(e_i_grad.shape)) is True:
            #     print("e_grad is 0 ")
            new_e_i = e_i.data - e_lr * e_i_grad
            new_e_i.requires_grad = True
            new_e_i.retain_grad()
            new_e_param.append(new_e_i)

            # if torch.equal(e_i, e_i_before) is True:
            #     e_i_num += 1

        # update stiefel
        # s_i_num=0
        new_s_params = []
        new_L_params = []
        new_R_params = []
        new_L_c = []
        new_R_c = []
        for j, s_i in enumerate(param['s_params'], 0):
            if s_i.grad is None:
                continue

            s_i_shape = s_i.shape
            # if torch.equal(s_i.grad, torch.zeros(s_i.grad.shape)) is True:
            #     print("s_grad is 0 ")
            M = s_i.data.view(s_i.shape[0], -1).T
            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            G = othogonal_projection(M, M_grad)
            n = G.shape[0]
            r = G.shape[1]
            GGt = G.mm(G.T).diag() / r
            GGt = torch.diag_embed(GGt)#.cuda()
            GtG = G.T.mm(G).diag() / n
            GtG = torch.diag_embed(GtG)#.cuda()
            L = self.beta * param['L_h0'][j] + (1 - self.beta) * GGt
            R = self.beta * param['R_h0'][j] + (1 - self.beta) * GtG
            L_c = torch.max(L, param['L_c0'][j])
            R_c = torch.max(R, param['R_c0'][j])
            L_p = torch.pow(L_c, -0.25).diag()
            L_p = torch.diag_embed(L_p)
            R_p = torch.pow(R_c, -0.25).diag()
            R_p = torch.diag_embed(R_p)
            G_p = L_p.mm(G).mm(R_p)
            s_i_be = s_i
            # s_i = my_retraction(M, G_p, s_lr)
            P = -s_lr * G_p
            P = othogonal_projection(M, P)
            s_i_p = retraction(M, P).T
            new_s_i = s_i_p.view(s_i_shape)

            new_s_i.requires_grad = True
            new_s_i.retain_grad()
            new_s_params.append(new_s_i)

            # update Lt Rt
            new_L_params.append(L)
            new_R_params.append(R)
            new_L_c.append(L_c)
            new_R_c.append(R_c)


        new_param = {
            'e_params': new_e_param,
            's_params': new_s_params,
            'bn_params': param['bn_params'],
            'L_h0': new_L_params,
            'L_c0': new_L_c,
            'R_h0': new_R_params,
            'R_c0': new_R_c
        }
        return new_param


    def zero_grad(self, param):
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            else:
                e_i.grad.zero_()
        for s_i in param['s_params']:
            if s_i.grad is None:
                continue
            else :
                s_i.grad.zero_()












