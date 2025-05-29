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

def parallel_transport(Q, W):
    return othogonal_projection(Q, W)

# use L_h0 to store e_momentum, use R_h0 to store s_momentum
class RSGDM(object):
    def __init__(self, opt):
        super(RSGDM, self).__init__()
        self.weight_decay = opt.weight_decay
        self.gama_e = opt.momentum_gama_e
        self.lr_e = opt.momentum_lr_e
        self.gama_s = opt.momentum_gama_s
        self.lr_s = opt.momentum_lr_s

    def step(self, param, e_lr, s_lr):
        # update e_params
        new_e_param = []
        new_L_h0 = []
        for j, e_i in enumerate(param['e_params'], 0):
            if e_i.grad is None:
                continue
            e_m = param['L_h0'][j]
            e_i_grad = e_i.grad
            e_i_grad = e_i_grad.add(e_i.data, alpha=self.weight_decay)
            new_e_m = self.gama_e * e_m + (1-self.gama_e) * e_i_grad
            new_e_i = e_i.data - self.lr_e * new_e_m
            new_e_i.requires_grad = True
            new_e_i.retain_grad()
            new_e_param.append(new_e_i)
            new_L_h0.append(new_e_m)

        # update s_param
        new_s_param = []
        new_R_h0 = []
        for j, s_i in enumerate(param['s_params'], 0):
            if s_i.grad is None:
                continue

        s_i_shape = s_i.shape
        M = s_i.data.view(s_i.data.shape[0], -1).T
        M_grad = s_i.grad.view(s_i.grad.shape[0], -1).T
        G = othogonal_projection(M, M_grad)
        s_m = param['R_h0'][j] # 和M_gard同型
        s_m = parallel_transport(M, s_m)
        new_s_m = self.gama_s * s_m + (1-self.gama_s) * G
        P = -self.lr_s * new_s_m
        s_i = retraction(M, P).T
        new_s_i = s_i.view(s_i_shape)
        new_s_i.requires_grad = True
        new_s_i.retain_grad()
        new_s_param.append(new_s_i)
        new_R_h0.append(new_s_m)

        new_param = {
            'e_params': new_e_param,
            's_params': new_s_param,
            'bn_params': param['bn_params'],
            'L_h0': new_L_h0,
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



