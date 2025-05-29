import torch
import torch.nn as nn
from retraction import Retraction

my_retraction = Retraction(1)

def othogonal_projection(M, M_grad):
    A = M.T.mm(M_grad)

    new = M_grad - M.mm(A)
    return new

retraction=Retraction(1)



class meta_optimizer(nn.Module):
    def __init__(self, opt):
        super(meta_optimizer, self).__init__()
        self.opt = opt
        self.weight_decay = opt.weight_decay
        self.lstm_layers = opt.lstm_layers
        self.hidden_size = opt.hidden_size
        self.lstm= nn.LSTM(1, hidden_size=self.hidden_size, num_layers=self.lstm_layers).cuda()
        self.output_size = 1
        self.L_linear = nn.Linear(self.hidden_size, self.output_size).cuda()
        self.R_linear = nn.Linear(self.hidden_size, self.output_size).cuda()
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
        #clamp_sign = torch.clamp(torch.exp(torch.Tensorparams['L_h0'](p)) * gradients, min=-1.0, max=1.0)
        return clamp_log.diag().view(1, -1, 1)  # 鍦╣radients鐨勬渶鍚庝竴缁磇nput_dims鎷兼帴

    def step(self, s_lr, s_i,M_grad,state):
        new_e_params = []
        # euc_update
        s_i_shape = s_i.shape
        # M_grad = s_i.grad
        M_grad = M_grad.squeeze().cuda()

        M = s_i.squeeze().cuda()
        Mshape = M.shape
        G = othogonal_projection(M, M_grad)
        G=G*s_lr
        GGt = G.mm(G.T)
        L_input = self.LogAndSign_Preprocess_Gradient(GGt)
        GtG = G.T.mm(G)
        R_input = self.LogAndSign_Preprocess_Gradient(GtG)

        L_h0_i = state[0].squeeze() .cuda() # cuda()
        L_c0_i = state[1].squeeze().cuda() # cuda()
        R_h0_i = state[2].squeeze() .cuda() # cuda()
        R_c0_i = state[3].squeeze().cuda()
        L_before = state[5].squeeze().cuda()  # cuda()
        R_before = state[4].squeeze() .cuda() # cuda()

        l_output, (hn_l, cn_l) = self.lstm(L_input, (L_h0_i, L_c0_i))
        r_output, (hn_r, cn_r) = self.lstm(R_input, (R_h0_i, R_c0_i))
        l_ = self.L_linear(l_output.squeeze()) * self.output_scale  # 鍥犱负LSTM鐨勮緭鍑烘槸褰撳墠姝ョ殑Hidden锛岄渶瑕佸彉鎹㈠埌output鐨勭浉鍚屽舰鐘朵笂
        l_ = l_ - l_.mean() + 1.0
        L = l_.squeeze().diag()
        # L = torch.diag_embed(L)
        r_ = self.R_linear(r_output.squeeze()) * self.output_scale
        r_ = r_ - r_.mean() + 1.0
        R = r_.squeeze().diag()
        # R = torch.diag_embed(R)
        # state[0]=hn_l.detach()
        # state[1]=hn_l.detach()
        # state[2]=hn_l.detach()
        # state[3]=cn_r.detach()
        # L = torch.pow(L, -0.25).diag()
        # R = torch.pow(R, -0.25).diag()
        L = torch.max(L, L_before)
        R = torch.max(R, R_before)

        # state[5]=L
        # state[4]=R
        next_state0 = hn_l.detach()
        next_state1 = cn_l.detach()
        next_state2 = hn_r.detach()
        next_state3 = cn_r.detach()
        next_state4 = R.detach()
        next_state5 = L.detach()

        P = L.mm(G).mm(R)
        # P = -s_lr*new_Gt
        P = othogonal_projection(M, P)
        P_p = P.unsqueeze(0)
        try:
            s_i_p = retraction(M, P_p,1)
        except:
            s_i_p = s_i
            print('svd')
            train_svd=True
        else:
            train_svd = False
            #s_i_p.requires_grad=True
            s_i_p.retain_grad()
        return train_svd, s_i_p, next_state0.unsqueeze(0), next_state1.unsqueeze(0), next_state2.unsqueeze(
                0), next_state3.unsqueeze(0), next_state4.unsqueeze(0), next_state5.unsqueeze(0)



    def zero_grad(self, s_i):
        if s_i.grad is None:
            return 0
        if s_i.grad.grad_fn is not None:
            s_i.grad.detach_()
        else:
            s_i.grad.requires_grad_(False)
            s_i.grad.zero_()















