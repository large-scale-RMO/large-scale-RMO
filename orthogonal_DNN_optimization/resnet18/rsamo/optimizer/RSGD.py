import torch
import torch.nn as nn

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


class RSGD(object):
    def __init__(self, opt):
        super(RSGD, self).__init__()
        self.decay = opt.weight_decay
        self.e_lr = opt.hand_e_lr
        self.s_lr = opt.hand_e_lr

    def step(self, param):
        new_param = []
        # update euc param
        new_e_param = []
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            e_i_grad = e_i.grad
            e_i_grad = e_i_grad.add(e_i.data, alpha= self.decay)
            e_i = e_i - self.e_lr*e_i_grad
            e_i.requires_grad= True
            e_i.retain_grad()
            new_e_param.append(e_i)

        # update stiefel param
        new_s_param = []
        for s_i in param['s_params']:
            if s_i.grad is None:
                continue


            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            M = s_i.data.view(s_i.shape[0], -1).T
            s_i_shape = s_i.shape
            P = -self.s_lr * M_grad
            s_i = retraction(M, P).T
            s_i = s_i.view(s_i_shape)
            s_i.requires_grad = True
            s_i.retain_grad()
            new_s_param.append(s_i)

        new_L_h0 = param['L_h0']
        new_L_c0 = param['L_c0']
        new_R_h0 = param['R_h0']
        new_R_c0 = param['R_c0']

        new_param = {
            'e_params': new_e_param,
            's_params': new_s_param,
            'L_h0': new_L_h0,
            'L-c0': new_L_c0,
            'R_h0': new_R_h0,
            'R_c0': new_R_c0
        }

        return new_param

    def zero_grad(self, param):
        for e_i in param['e_params']:
            e_i.grad.zero_()
        for s_i in param['s_params']:
            s_i.gard.zero_()









