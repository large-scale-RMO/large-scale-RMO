import torch
import torch.nn as nn
from Model.Retraction import Retraction
import torch.nn.functional as F
import numpy as np
import pdb
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from LSTM_Optimizee_Model0 import LSTM_Optimizee_Model0
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
        self.Neumann_series = opt.Neumann_series

        self.lstm_layers = opt.lstm_layers
        self.Neumann_alpha = opt.Neumann_alpha
        self.hidden_size = opt.hidden_size
        self.LSTM_Optimizee0 = LSTM_Optimizee_Model0(opt, 576, 64, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)
        print("gmLSTM 0 ready----")
        self.LSTM_Optimizee1 = LSTM_Optimizee_Model0(opt, 576, 128, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)
        print("gmLSTM 1 ready----")
        self.LSTM_Optimizee2 = LSTM_Optimizee_Model0(opt, 1152, 128, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)
        print("gmLSTM 2 ready----")
        self.LSTM_Optimizee3 = LSTM_Optimizee_Model0(opt, 1152, 256, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 3 ready----")
        self.LSTM_Optimizee4 = LSTM_Optimizee_Model(opt, 2304, 256, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 4 ready----")

        self.LSTM_Optimizee5 = LSTM_Optimizee_Model(opt, 2304, 512, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 5 ready----")
        self.LSTM_Optimizee6 = LSTM_Optimizee_Model(opt, 4608, 512, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 6 ready----")
        self.adam_optimizer0 = torch.optim.Adam(self.LSTM_Optimizee0.parameters(), lr=opt.adam_lr)
        self.adam_optimizer1 = torch.optim.Adam(self.LSTM_Optimizee1.parameters(), lr=opt.adam_lr)
        self.adam_optimizer2 = torch.optim.Adam(self.LSTM_Optimizee2.parameters(), lr=opt.adam_lr)
        self.adam_optimizer3 = torch.optim.Adam(self.LSTM_Optimizee3.parameters(), lr=opt.adam_lr)
        self.adam_optimizer4 = torch.optim.Adam(self.LSTM_Optimizee4.parameters(), lr=opt.adam_lr)
        self.adam_optimizer5 = torch.optim.Adam(self.LSTM_Optimizee5.parameters(), lr=opt.adam_lr)
        self.adam_optimizer6 = torch.optim.Adam(self.LSTM_Optimizee6.parameters(), lr=opt.adam_lr)

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
        self.loss_func = nn.CrossEntropyLoss()

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

    def last_step(self, params, param0, e_lr, s_lr, inputs, labels, model):
        new_e_params = []
        # euc_update
        break_flag = False

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

            s_i_shape = s_i.shape
            M_grad = s_i.grad.view(s_i.shape[0], -1).T

            G = M_grad.data.requires_grad_()

            M = s_i.data.view(s_i.shape[0], -1).T
            M = M.requires_grad_()
            r = M.shape[0]
            c = M.shape[1]
            Mshape = M.shape
            G = othogonal_projection(M, M_grad)
            G = G.unsqueeze(0).to("cuda:0")
            M_ = M.unsqueeze(0)

            # print(params['state'][j])
            state_i = params['state'][j]

            # print(state_i)

            if j == 0:
                optimizee = self.LSTM_Optimizee0
                adam_optimizer = self.adam_optimizer0
                lr, update, state = self.LSTM_Optimizee0(G, state_i)

            elif j == 1:

                optimizee = self.LSTM_Optimizee1
                adam_optimizer = self.adam_optimizer1
                lr, update, state = self.LSTM_Optimizee1(G, state_i)

            elif j == 2:
                optimizee = self.LSTM_Optimizee2
                adam_optimizer = self.adam_optimizer2

                lr, update, state = self.LSTM_Optimizee2(G, state_i)

            elif j == 3:
                optimizee = self.LSTM_Optimizee3
                adam_optimizer = self.adam_optimizer3
                lr, update, state = self.LSTM_Optimizee3(G, state_i)

            elif j == 4 or j == 5:
                optimizee = self.LSTM_Optimizee4
                adam_optimizer = self.adam_optimizer4

                lr, update, state = self.LSTM_Optimizee4(G, state_i)

            elif j == 6:
                optimizee = self.LSTM_Optimizee5
                adam_optimizer = self.adam_optimizer5
                lr, update, state = self.LSTM_Optimizee5(G, state_i)

            elif j == 7 or j == 8 or j == 9 or j == 10:
                optimizee = self.LSTM_Optimizee6
                adam_optimizer = self.adam_optimizer6
                lr, update, state = self.LSTM_Optimizee6(G, state_i)

            lr = torch.abs(lr) * s_lr

            update_R = -lr.to("cuda:0") * (update.to("cuda:0") + G.to("cuda:0")).requires_grad_()

            # M.grad.data.zero_()
            update_R = update_R.squeeze()
            M_end = retraction(M, update_R).requires_grad_()
            if state[0] is None:
                state = (state[0], state[1], state[2], state[3])
            else:
                state = (state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach())

            vector = param0['s_params'][j].grad.view(s_i.shape[0], -1).T.detach()
            Jacobi_vector = vector
            vector_temp_list = []
            for i in range(1):
                vector_temp = torch.autograd.grad(M_end, update_R, grad_outputs=vector, retain_graph=True)[
                    0]  # dm/dy   其中y是更新向量，m是更新之后的参数
                # print('vector_temp111111:{}'.format(torch.sum(vector_temp)))
                vector_temp_list.append(vector_temp.detach().cpu().numpy())

                vector_temp = torch.autograd.grad(update_R, G, grad_outputs=vector_temp, retain_graph=True)[0]
                vector_temp2 = torch.autograd.grad(M_end, M, grad_outputs=vector, retain_graph=True)[0]

                inputs_clone, labels_clone = inputs.clone().cuda(), labels.clone().cuda()
                # M = params['s_params'][j].data.view(s_i.shape[0], -1).T.requires_grad_()
                M = params['s_params'][j]
                outputs = model(inputs_clone, params)
                hessan_loss = self.loss_func(outputs, labels_clone)
                # hessan_loss.backward(retain_graph=True)
                grad_M = torch.autograd.grad(hessan_loss, M, retain_graph=True, create_graph=True)[0]
                # M_t = params['s_params'][j].data.view(s_i.shape[0], -1).T
                grad_M = grad_M.view(s_i.shape[0], -1).T
                # grad_M.requires_grad=True
                # M_t = param1['s_params'][j]
                vec = vector_temp
                h = torch.sum(grad_M * vec)
                vector_temp = torch.autograd.grad(h, M)[0].view(s_i.shape[0], -1).T
                vector = vector - self.Neumann_alpha * (vector_temp + vector_temp2)
                Jacobi_vector = Jacobi_vector + vector
            Jacobi_vector_sum = torch.sum(Jacobi_vector).detach().cpu().numpy().tolist()
            if np.isnan(Jacobi_vector_sum):
                vector_temp_len = len(vector_temp_list)
                for len_i in range(vector_temp_len):
                    print(vector_temp_list[i])
                print('ERROR NAN')
                break_flag = True
                break
            vector_M_update = torch.autograd.grad(M_end, update_R, grad_outputs=Jacobi_vector.squeeze())[0]
            update_R_optimizee = torch.autograd.grad(update_R, optimizee.parameters(), grad_outputs=vector_M_update)

            check = 0
            for p in optimizee.parameters():
                check = check + 1 if type(p.grad) == type(None) else check
            if check > 0:
                print('-------------------------------------')
                optimizer_param_sum = 0

                for param_i in optimizee.parameters():
                    optimizer_param_sum = optimizer_param_sum + torch.norm(param_i.to("cuda:0"))
                # initialize the grad fields properly
                # output = f(inputs,M)
                # output=output.permute(0,2,1)
                # output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )

                # back_loss = criterion(output, labels)
                back_loss = optimizer_param_sum
                back_loss.backward()  # this would initialize required variables

            for i, p in enumerate(optimizee.parameters()):
                p.grad = update_R_optimizee[i]

            adam_optimizer.step()
            M.to('cpu')
            # M.grad = None
            M = None
            # del M.grad
            del M
            vector.to('cpu')
            # vector.grad = None
            vector = None
            # del vector.grad
            del vector
            grad_M.to('cpu')
            # grad_M.grad = None
            grad_M = None
            del grad_M
            # del grad_M.grad
            M_end.to('cpu')
            # M_end.grad = None
            M_end = None
            del M_end
            # del M_end.grad
            G.to('cpu')
            # G.grad = None
            G = None
            del G
            # del G.grad

            update_R.to('cpu')
            # update_R.grad = None
            update_R = None
            del update_R
            # del update_R.grad
            vector_temp.to('cpu')
            # vector_temp.grad = None
            vector_temp = None
            del vector_temp
            # del vector_temp.grad
            vector_temp2.to('cpu')
            # vector_temp2.grad = None
            vector_temp2 = None
            del vector_temp2
            # del vector_temp2.grad
            Jacobi_vector.to('cpu')
            # Jacobi_vector.grad = None
            Jacobi_vector = None
            del Jacobi_vector
            # del Jacobi_vector.grad
            vector_M_update.to('cpu')
            # vector_M_update.grad = None
            vector_M_update = None
            del vector_M_update
            # del vector_M_update.grad
            # update_R_optimizee.to('cpu')
            # update_R_optimizee.grad = None
            update_R_optimizee = None
            del update_R_optimizee
            # del update_R_optimizee.grad

            del M_grad, update, state, vec, outputs, hessan_loss, h, Jacobi_vector_sum, optimizer_param_sum, back_loss
            # del update_R_optimizee
            torch.cuda.empty_cache()

        return break_flag

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

            if j == 0:
                lr, update, state = self.LSTM_Optimizee0(G, state_i)

            elif j == 1:
                lr, update, state = self.LSTM_Optimizee1(G, state_i)

            elif j == 2:
                lr, update, state = self.LSTM_Optimizee2(G, state_i)

            elif j == 3:
                lr, update, state = self.LSTM_Optimizee3(G, state_i)

            elif j == 4 or j == 5:
                lr, update, state = self.LSTM_Optimizee4(G, state_i)

            elif j == 6:
                lr, update, state = self.LSTM_Optimizee5(G, state_i)

            elif j == 7 or j == 8 or j == 9 or j == 10:
                lr, update, state = self.LSTM_Optimizee6(G, state_i)

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

















