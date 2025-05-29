import torch
import torch.nn as nn
from Model.Retraction import Retraction
my_Retraction = Retraction(1)

def othogonal_projection(M, M_grad):
    A = M.transpose(0,1).mm(M_grad)
    A_sym = 0.5*(A + A.T)
    new = M_grad - M.mm(A_sym)
    return new

def retraction(M, P):
    A = M + P
    Q, R = A.qr()
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out

def parallel_transport(Q, W):
    return othogonal_projection(Q, W)


class RSVRG(object):
    def __init__(self, opt):
        super(RSVRG, self).__init__()
        self.weight_decay = opt.weight_decay


    def step(self, param_p, param, s_grad, e_lr, s_lr):
        # update ecu params
        new_e_param = []
        for e_i in param_p['e_params']:
            if e_i.grad is None:
                continue
            e_grad = e_i.grad
            e_grad = e_grad.add_(e_i.data, alpha=self.weight_decay)
            new_e_i = e_i.data - e_lr * e_grad
            new_e_i.requires_grad = True
            new_e_i.retain_grad()
            new_e_param.append(new_e_i)

        # update stiefel params
        new_s_param = []
        for j, s_i in enumerate(param_p['s_params'], 0):
            if s_i.grad is None:
                continue
            if j==0:
                s_i_g = s_i.grad
                if self.weight_decay != 0:
                    s_i_g = s_i_g.add(s_i.data, alpha=self.weight_decay)
                new_s_i = s_i.data - e_lr* s_i_g
                new_s_i.requires_grad=True
                new_s_i.retain_grad()
                new_s_param.append(new_s_i)
                continue
            s_grad_0 = param['s_params'][j].grad
            if s_grad_0 is None:
                continue
            s_grad_0_p = s_grad_0 - s_grad[j]
            s_grad_0_p = s_grad_0_p.view(s_grad_0_p.shape[0], -1).T
            s_i_shape = s_i.shape
            M = s_i.data.view(s_i.shape[0], -1).T
            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            s_i_update = parallel_transport(M, s_grad_0_p)
            v = M_grad - s_i_update
            P = - s_lr * v
            s_i = retraction(M, P).T
            new_s_i = s_i.view(s_i_shape)
            new_s_i.requires_grad = True
            new_s_i.retain_grad()
            new_s_param.append(new_s_i)

        new_params = {
            'e_params': new_e_param,
            's_params': new_s_param,
            'bn_params': param_p['bn_params'],
            'L_h0': [],
            'L_c0': [],
            'R_h0': [],
            'R_c0': []
        }

        return new_params

    def zero_grad(self, param):
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            else:
                e_i.grad.zero_()
        for s_i in param['s_params']:
            if s_i.grad is None:
                continue
            else:
                s_i.grad.zero_()






