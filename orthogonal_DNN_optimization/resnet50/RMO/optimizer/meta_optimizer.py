import torch
import torch.nn as nn
from Model.Retraction import Retraction
import torch.nn.functional as F
import pdb
from LSTM_Optimizee_Model import LSTM_Optimizee_Model

my_retraction = Retraction(1)


def othogonal_projection(M, M_grad):
    A = M.transpose(0, 1).mm(M_grad)
    A_sym = 0.5 * (A + A.T)
    new = M_grad - M.mm(A_sym)
    return new


def retraction(M, P):
    A = M + P
    Q, R = torch.qr(A)
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out


class meta_optimizer(nn.Module):
    def __init__(self, opt):
        super(meta_optimizer, self).__init__()
        self.opt = opt
        self.weight_decay = opt.weight_decay
        self.lstm_layers = opt.lstm_layers
        self.hidden_size = opt.hidden_size
        self.LSTM_Optimizee0 = LSTM_Optimizee_Model(opt, 576, 64, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)
        print("gmLSTM 0 ready----")
        self.LSTM_Optimizee1 = LSTM_Optimizee_Model(opt, 256,64, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)
        print("gmLSTM 1 ready----")
        self.LSTM_Optimizee2 = LSTM_Optimizee_Model(opt, 256, 128, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)
        print("gmLSTM 2 ready----")
        self.LSTM_Optimizee3 = LSTM_Optimizee_Model(opt, 1152, 128, batchsize_data=opt.batchsize,
                                                   batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 3 ready----")
        self.LSTM_Optimizee4 = LSTM_Optimizee_Model(opt, 512, 128, batchsize_data=opt.batchsize,
                                                   batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 4 ready----")

        self.LSTM_Optimizee5 = LSTM_Optimizee_Model(opt, 512, 256, batchsize_data=opt.batchsize,
                                                   batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 5 ready----")
        self.LSTM_Optimizee6 = LSTM_Optimizee_Model(opt, 2304, 256, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 6 ready----")
        self.LSTM_Optimizee7 = LSTM_Optimizee_Model(opt, 1024, 256, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 7 ready----")
        self.LSTM_Optimizee8 = LSTM_Optimizee_Model(opt, 1024, 512, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 8 ready----")
        self.LSTM_Optimizee9 = LSTM_Optimizee_Model(opt, 4608, 512, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 9 ready----")
        self.LSTM_Optimizee10 = LSTM_Optimizee_Model(opt, 2048, 512, batchsize_data=opt.batchsize,
                                                  batchsize_para=opt.batchsize_param)#.cuda(1)
        print("gmLSTM 10 ready----")
        # self.LSTM_Optimizee7 = LSTM_Optimizee_Model(opt, 147, 64, batchsize_data=opt.batchsize,
        #                                             batchsize_para=opt.batchsize_param)
        # print("gmLSTM 7 ready----")
        #
        # self.proj0 = nn.Linear(576*64, 1024).cuda()
        # print("project 0 ready----")
        # self.proj1 = nn.Linear(576*128, 1024).cuda()
        # print("project 1 ready----")
        # self.proj2 = nn.Linear(128*1152, 1024).cuda()
        # print("project 2 ready----")
        # self.proj3 = nn.Linear(256*1152, 1024).cuda()
        # print("project 3 ready----")
        # self.proj4 = nn.Linear(256*2304, 1024).cuda()
        # print("project 4 ready----")
        # self.proj5 = nn.Linear(512*2304, 1024).cuda()
        # print("project 5 ready----")
        #
        # self.proj7 = nn.Linear(4608 * 512, 1024).cuda()
        # print("project 6 ready----")
        # #
        # self.proj_2 = nn.Linear(1024, 1).cuda()
        self.p = opt.p
        self.output_scale = opt.output_scale

    def LogAndSign_Preprocess_Gradient(self, gradients):
        """
        Args:
          gradients: `Tensor` of gradients with shape `[d_1, ..., d_n]`.
          p       : `p` > 0 is a parameter controlling how small gradients are disregarded
        Returns:
          `Tensor` with shape `[d_1, ..., d_n-1, 2 * d_n]`. The first `d_n` elements
          along the nth dimension correspond to the `log output` \in [-1,1] and the remaining
          `d_n` elements to the `sign output`.
        """
        p = self.p
        log = torch.log(torch.abs(gradients)).cuda()
        clamp_log = torch.clamp(log / p, min=-1.0, max=1.0).cuda()
        # clamp_sign = torch.clamp(torch.exp(torch.Tensor(p)) * gradients, min=-1.0, max=1.0)
        return clamp_log.diag().view(1, -1, 1)  # 在gradients的最后一维input_dims拼接

    def step(self, params, e_lr, s_lr):
        new_e_params = []
        # euc_update
        for e_i in params['e_params']:
            if e_i.grad is None:
                continue
            e_i_g = e_i.grad
            if self.weight_decay != 0:
                e_i_g = e_i_g.add(e_i.data, alpha=self.weight_decay)
            new_e_i = e_i.data - e_lr * e_i_g
            new_e_i.requires_grad = True
            new_e_i.retain_grad()
            new_e_params.append(new_e_i)

        # stief_update
        new_s_params = []
        new_state = []
        for j, s_i in enumerate(params['s_params']):
            if s_i.grad is None:
                continue
            #pdb.set_trace()
            if j==0 or j==1 or j==3 or j==4 or j==7 or j==10 or j==13 or j==14 or j==17 or j==20 or j==23 or j==26 or j==27 or j==30 or j==33 or j==36 or j==39 or j==42 or j==45 or j==46 or j==49 or j==52:
                s_i_g = s_i.grad
                if self.weight_decay != 0:
                    s_i_g = s_i_g.add(s_i.data, alpha=self.weight_decay)
                new_s_i = s_i.data - e_lr* s_i_g
                new_s_i.requires_grad=True
                new_s_i.retain_grad()
                new_s_params.append(new_s_i)
                state_i = None
                new_state.append(state_i)
                continue

            s_i_shape = s_i.shape
            M_grad = s_i.grad.view(s_i.shape[0], -1).T
            G = M_grad
            M = s_i.data.view(s_i.shape[0], -1).T
            r = M.shape[0]
            c = M.shape[1]
            Mshape = M.shape
            G = othogonal_projection(M, M_grad)
            G = G.unsqueeze(0).to("cuda:0")
            M_ = M.unsqueeze(0)
            # pdb.set_trace()
            # print(params['state'][j])
            state_i = params['state'][j]

            # print(state_i)

            # if j == 0:
            #     lr, update, state = self.LSTM_Optimizee7(G, state_i)

            if j == 2 or j==6 or j==9:
                lr, update, state= self.LSTM_Optimizee0(G, state_i)

            elif j == 5 or j==8:
                lr, update, state = self.LSTM_Optimizee1(G, state_i)

            elif j == 11:
                lr, update, state= self.LSTM_Optimizee2(G, state_i)

            elif j == 12 or j == 16 or j==19 or j==22:
                lr, update, state = self.LSTM_Optimizee3(G, state_i)

            elif j == 18 or j==15 or j==21:
                lr, update,state = self.LSTM_Optimizee4(G, state_i)
            elif j == 24 :
                lr, update, state = self.LSTM_Optimizee5(G, state_i)
            elif j == 25 or j==29 or j==32 or j == 35 or j==38 or j==41:
                lr, update, state = self.LSTM_Optimizee6(G, state_i)
            elif j == 28 or j==31 or j==34 or j == 37 or j==40:
                lr, update, state = self.LSTM_Optimizee7(G, state_i)
            elif j == 43:
                lr, update, state = self.LSTM_Optimizee8(G, state_i)
            elif j == 44 or j == 48 or j == 51:
                lr, update, state = self.LSTM_Optimizee9(G, state_i)
            elif j == 47 or j == 50:
                lr, update, state = self.LSTM_Optimizee10(G, state_i)

            lr = torch.abs(lr) * s_lr


            update_R = -lr.to("cuda:0") * (update.to("cuda:0") + G.to("cuda:0"))
            # M.grad.data.zero_()

            M_new = retraction(M, update_R.squeeze()).T
            if state[0] is None:
                state = (state[0], state[1], state[2], state[3])
            else:
                state = (state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach())
            M_new = M_new.detach()
            M_new = M_new.squeeze()
            M_new = M_new.view(s_i_shape)
            M_new.requires_grad = True
            M_new.retain_grad()
            #
            new_s_params.append(M_new)
            new_state.append(state)

        # update
        new_params = {
            'e_params': new_e_params,
            's_params': new_s_params,
            'bn_params': params['bn_params'],
            'state': new_state
        }

        return new_params

    def zero_grad(self, param):
        for e_i in param['e_params']:
            if e_i.grad is None:
                continue
            if e_i.grad.grad_fn is not None:
                e_i.grad.detach_()
            else:
                e_i.grad.requires_grad_(False)
            e_i.grad.zero_()

        for s_i in param['s_params']:
            if s_i.grad is None:
                continue
            if s_i.grad.grad_fn is not None:
                s_i.grad.detach_()
            else:
                s_i.grad.requires_grad_(False)
            s_i.grad.zero_()

















