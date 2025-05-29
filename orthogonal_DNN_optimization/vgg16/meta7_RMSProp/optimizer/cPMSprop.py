import torch
import torch.nn as nn
from Model.mysqrt import M_Sqrt



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


class cRMSProp(object):
    def __init__(self, opt):
        super(cRMSProp, self).__init__()
        self.decay = opt.weight_decay
        self.e = 0.0000001
        self.beta = opt.rmsprop_beta
        self.sqrt = M_Sqrt(1)


    def step(self, param, e_lr, s_lr):
        new_e_param = []
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            e_i_grad = e_i.grad
            e_i_grad = e_i_grad.add_(e_i.data, alpha=self.decay)
            new_e = e_i.data - e_lr * e_i_grad
            new_e.requires_grad = True
            new_e.retain_grad()
            new_e_param.append(new_e)

        new_s_param = []
        new_R_h0 = []
        for j, s_i in enumerate(param['s_params'], 0):
            if s_i.grad is None:
                continue
            s_i_shape = s_i.shape
            M = s_i.data.view(s_i.shape[0], -1).T
            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            M_grad_P = torch.mul(M_grad, M_grad)
            G = othogonal_projection(M, M_grad)
            G_P = othogonal_projection(M, M_grad_P)
            m = param['R_h0'][j].cuda()
            m_p = parallel_transport(M, m)
            new_m = self.beta * m_p + (1 - self.beta) * G_P
            new_R_h0.append(new_m)
            one = torch.ones(new_m.shape).cuda()
            e = self.e * one

            m_sqrt = self.sqrt(new_m)
            m_sqrt_p = m_sqrt.data + e
            new_lr = s_lr / m_sqrt_p
            P = -new_lr * G
            s_i = retraction(M, P).T
            new_s_i = s_i.view(s_i_shape)
            new_s_i.requires_grad = True
            new_s_i.retain_grad()
            new_s_param.append(new_s_i)

        new_param = {
            'e_params': new_e_param,
            's_params': new_s_param,
            'bn_params': param['bn_params'],
            'L_h0': [],
            'L_c0': [],
            'R_h0': new_R_h0,
            'R_c0': []
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
            else:
                s_i.grad.zero_()




