import torch
import torch.nn as nn
from Model.Retraction import Retraction

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
        # self.e_lr = opt.hand_e_lr
        # self.s_lr = opt.hand_e_lr

    def step(self, param, e_lr, s_lr):

        # update euc param
        e_i_num = 0
        for e_i in param['e_params']:
            if e_i.grad is None:
                # print('e_grad_none')
                continue
            e_i_grad = e_i.grad
            e_i_be = e_i
            # e_i.data.add_(e_i_grad, alpha=-e_lr)
            e_i = e_i.data - e_lr * e_i_grad
            if torch.equal(e_i_be, e_i) is True:
                e_i_num += 1
        if e_i_num == len(param['e_params']):
            print("e doesnt change------------")


        # update stiefel param
        s_i_num = 0
        for s_i in param['s_params']:
            if s_i.grad is None:
                # print('s_grad_none')
                continue

            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            M = s_i.data.view(s_i.shape[0], -1).T
            s_i_shape = s_i.shape
            # P = -s_lr * M_grad
            # s_i = retraction(M, P).T
            my_Retraction = Retraction(1)
            s_i_before = s_i
            s_i = my_Retraction(M, M_grad, s_lr)
            s_i = s_i.view(s_i_shape)
            if torch.equal(s_i_before, s_i) is True:
                s_i_num += 1
        if s_i_num == len(param['s_params']) :
            print('s doesnt change ------------------')



    def zero_grad(self, param):
        for e_i in param['e_params']:
            e_i.grad.zero_()
        for s_i in param['s_params']:
            s_i.gard.zero_()









