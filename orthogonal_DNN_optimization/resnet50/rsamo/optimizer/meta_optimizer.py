
import torch
import torch.nn as nn
from retraction import Retraction

my_retraction = Retraction(1)

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


class meta_optimizer(nn.Module):
    def __init__(self, opt):
        super(meta_optimizer, self).__init__()
        self.opt = opt
        self.weight_decay = opt.weight_decay
        self.lstm_layers = opt.lstm_layers
        self.hidden_size = opt.hidden_size
        self.lstm= nn.LSTM(1, hidden_size=self.hidden_size, num_layers=self.lstm_layers).cuda()#.to("cuda:2)
        self.output_size = 1
        self.L_linear = nn.Linear(self.hidden_size, self.output_size).cuda()#to("cuda:2")
        self.R_linear = nn.Linear(self.hidden_size, self.output_size).cuda()#.to("cuda:2")
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
        log = torch.log(torch.abs(gradients)).cuda()#.to("cuda:2")
        clamp_log = torch.clamp(log / p, min=-1.0, max=1.0).cuda()#.to("cuda:2")
        #clamp_sign = torch.clamp(torch.exp(torch.Tensor(p)) * gradients, min=-1.0, max=1.0)
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
            new_e_i = e_i.data - e_lr*e_i_g
            new_e_i.requires_grad = True
            new_e_i.retain_grad()
            new_e_params.append(new_e_i)

        # stief_update
        new_s_params = []
        new_L_h0 = []
        new_L_c0 = []
        new_R_h0 = []
        new_R_c0 = []
        new_L_before = []
        new_R_before = []
        for j, s_i in enumerate(params['s_params']):
            if s_i.grad is None:
                continue
            if j==0 or j==1 or j==3 or j==4 or j==7 or j==10 or j==13 or j==14 or j==17 or j==20 or j==23 or j==26 or j==27 or j==30 or j==33 or j==36 or j==39 or j==42 or j==45 or j==46 or j==49 or j==50 or j==52 :
                s_i_g = s_i.grad
                if self.weight_decay != 0:
                    s_i_g = s_i_g.add(s_i.data, alpha=self.weight_decay)
                new_s_i = s_i.data - e_lr* s_i_g
                new_s_i.requires_grad=True
                new_s_i.retain_grad()
                new_s_params.append(new_s_i)

                L_h0_i = params['L_h0'][j].cuda()#.to("cuda:2")
                L_c0_i = params['L_c0'][j].cuda()#.to("cuda:2")
                R_h0_i = params['R_h0'][j].cuda()#to("cuda:2")
                R_c0_i = params['R_c0'][j].cuda()#to("cuda:2")
                L_before = params['L_before'][j].cuda()#to("cuda:2")
                R_before = params['R_before'][j].cuda()#to("cuda:2")
                new_L_h0.append(L_h0_i.detach())
                new_L_c0.append(L_c0_i.detach())
                new_R_h0.append(R_h0_i.detach())
                new_R_c0.append(R_c0_i.detach())
                new_L_before.append(params['L_before'][j].detach())
                new_R_before.append(params['R_before'][j].detach())
                continue

            s_i_shape = s_i.shape
            M_grad = s_i.grad.view(s_i.shape[0],-1).T.cuda()
            M_grad_p =  s_lr*M_grad
            M = s_i.data.view(s_i.shape[0],-1).T.cuda()
            r = M.shape[0]
            c = M.shape[1]
            Mshape = M.shape
            G = othogonal_projection(M, M_grad_p)
            GGt = G.mm(G.T)/c
            L_input= self.LogAndSign_Preprocess_Gradient(GGt)
            GtG = G.T.mm(G)/r
            R_input = self.LogAndSign_Preprocess_Gradient(GtG)

            L_h0_i = params['L_h0'][j].cuda()#.to("cuda:2")
            L_c0_i = params['L_c0'][j].cuda()#to("cuda:2")
            R_h0_i = params['R_h0'][j].cuda()#to("cuda:2")
            R_c0_i = params['R_c0'][j].cuda()#to("cuda:2")
            L_before = params['L_before'][j].cuda()#.to("cuda:2")
            R_before = params['R_before'][j].cuda()#to("cuda:2")


            l_output, (hn_l, cn_l) = self.lstm(L_input, (L_h0_i, L_c0_i))
            r_output, (hn_r, cn_r) = self.lstm(R_input, (R_h0_i, R_c0_i))
            l_ = self.L_linear(l_output.squeeze())*self.output_scale #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上
            l_ = l_ - l_.mean() + 1.0
            L = l_.squeeze().diag()
            # L = torch.diag_embed(L)
            r_ = self.R_linear(r_output.squeeze())*self.output_scale
            r_ = r_ - r_.mean() + 1.0
            R = r_.squeeze().diag()
            # R = torch.diag_embed(R)
            new_L_h0.append(hn_l.detach())
            new_L_c0.append(cn_l.detach())
            new_R_h0.append(hn_r.detach())
            new_R_c0.append(cn_r.detach())
            #L = torch.pow(L, -0.25).diag()
            #R = torch.pow(R, -0.25).diag()
            L = torch.max(L, L_before)
            R = torch.max(R, R_before)

            new_L_before.append(L)#.to("cuda:1"))
            new_R_before.append(R)#.to("cuda:1"))

            P = L.mm(G).mm(R)
            # P = -s_lr*new_Gt
            P = othogonal_projection(M, P)

            #s_i_p = my_retraction(M.unsqueeze(0), P.unsqueeze(0),1).squeeze().T
            s_i_p = retraction(M, -P).T
            new_s_i = s_i_p.view(s_i_shape)
            # new_s_i.requires_grad = True
            new_s_i.retain_grad()
            new_s_params.append(new_s_i)#.to("cuda:1"))

        # update
        new_params = {
            'e_params': new_e_params,
            's_params': new_s_params,
            'bn_params': params['bn_params'],
            'L_h0': new_L_h0,
            'L_c0': new_L_c0,
            'R_h0': new_R_h0,
            'R_c0': new_R_c0,
            'L_before': new_L_before,
            'R_before': new_R_before
        }

        return new_params



















