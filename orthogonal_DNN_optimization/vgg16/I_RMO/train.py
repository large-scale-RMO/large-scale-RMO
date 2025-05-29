import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
import torch

import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import config
import re

# from Model.vgg16 import vgg16
from Model.vgg16_BN import vgg16_BN
from Model.Cutout import Cutout
# from Model.AutoAugment import AutoAugment, Cutout
# from Model.resnet50 import Resnet50
# from Model.resnet18 import Resnet18
from optimizer.RSGD import RSGD
# from optimizer.meta_optimizer import meta_optimizer
from RB import RB
from tensorboardX import SummaryWriter
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp
import numpy as np
import time

import torch
import torch.nn as nn
from Model.Retraction import Retraction
import torch.nn.functional as F
import numpy as np
import pdb
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from LSTM_Optimizee_Model0 import LSTM_Optimizee_Model0
from LSTM_Optimizee_Model1 import LSTM_Optimizee_Model1

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
        self.LSTM_Optimizee4 = LSTM_Optimizee_Model0(opt, 2304, 256, batchsize_data=opt.batchsize,
                                                     batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 4 ready----")

        self.LSTM_Optimizee5 = LSTM_Optimizee_Model0(opt, 2304, 512, batchsize_data=opt.batchsize,
                                                     batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 5 ready----")
        self.LSTM_Optimizee6 = LSTM_Optimizee_Model(opt, 4608, 512, batchsize_data=opt.batchsize,
                                                    batchsize_para=opt.batchsize_param)  # .cuda(1)
        print("gmLSTM 6 ready----")
        self.adam_optimizer0 = torch.optim.SGD(self.LSTM_Optimizee0.parameters(), lr=opt.adam_lr)
        self.adam_optimizer1 = torch.optim.SGD(self.LSTM_Optimizee1.parameters(), lr=opt.adam_lr)
        self.adam_optimizer2 = torch.optim.SGD(self.LSTM_Optimizee2.parameters(), lr=opt.adam_lr)
        self.adam_optimizer3 = torch.optim.SGD(self.LSTM_Optimizee3.parameters(), lr=opt.adam_lr)
        self.adam_optimizer4 = torch.optim.SGD(self.LSTM_Optimizee4.parameters(), lr=opt.adam_lr)
        self.adam_optimizer5 = torch.optim.SGD(self.LSTM_Optimizee5.parameters(), lr=opt.adam_lr)
        self.adam_optimizer6 = torch.optim.SGD(self.LSTM_Optimizee6.parameters(), lr=opt.adam_lr)

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
        return clamp_log.diag().view(1, -1, 1)

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
            print(len(params['s_params']))
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
            G = G.unsqueeze(0).to("cuda:2")
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

            elif j == 4 or j==5:
                optimizee = self.LSTM_Optimizee4
                adam_optimizer = self.adam_optimizer4

                lr, update, state = self.LSTM_Optimizee4(G, state_i)



            elif j == 6:
                optimizee = self.LSTM_Optimizee5
                adam_optimizer = self.adam_optimizer5
                lr, update, state = self.LSTM_Optimizee5(G, state_i)

            elif j == 7 or j == 8 or j == 9 or j == 10 or j==11 or j==12:

                optimizee = self.LSTM_Optimizee6
                adam_optimizer = self.adam_optimizer6
                lr, update, state = self.LSTM_Optimizee6(G.to("cuda:2"), state_i)


            lr = torch.abs(lr) * s_lr

            update_R = -lr.to("cuda:2") * (update.to("cuda:2") + G.to("cuda:2")).requires_grad_()

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
                    0]
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
                    print(vector_temp_list[len_i])

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
                print(j)
                optimizer_param_sum = 0

                for param_i in optimizee.parameters():
                    optimizer_param_sum = optimizer_param_sum + torch.norm(param_i.to("cuda:2"))
                # initialize the grad fields properly
                # output = f(inputs,M)
                # output=output.permute(0,2,1)
                # output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )

                # back_loss = criterion(output, labels)
                back_loss = optimizer_param_sum
                adam_optimizer.zero_grad()
                back_loss.backward()  # this would initialize required variables

            for i, p in enumerate(optimizee.parameters()):
                p.grad = update_R_optimizee[i]

            adam_optimizer.step()
            adam_optimizer.zero_grad()

            for i, p in enumerate(optimizee.parameters()):
                p.grad = None
                del p.grad
            torch.cuda.empty_cache()

            del update_R_optimizee
            torch.cuda.empty_cache()

            # params['s_params'][j].grad.to('cpu')
            params['s_params'][j].grad = None
            del params['s_params'][j].grad
            torch.cuda.empty_cache()

            # adam_optimizer.to('cpu')
            del adam_optimizer
            # optimizee.to('cpu')
            del optimizee
            torch.cuda.empty_cache()

            # param0['s_params'][j].grad.to('cpu')
            param0['s_params'][j].grad = None
            del param0['s_params'][j].grad
            torch.cuda.empty_cache()

            del inputs_clone, labels_clone

            vector_temp_len = len(vector_temp_list)
            for len_i in range(vector_temp_len):
                del vector_temp_list[len_i]
            del vector_temp_list
            torch.cuda.empty_cache()

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

            # del update_R_optimizee.grad

            del M_grad, update, state, vec, outputs, hessan_loss, h, Jacobi_vector_sum, optimizer_param_sum, back_loss, state_i, lr
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
        for s_i in params['s_params']:
            print(s_i.shape)
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
            G = G.unsqueeze(0).to("cuda:2")
            M_ = M.unsqueeze(0)
            #
            # print(params['state'][j])
            state_i = params['state'][j]

            # print(state_i)

            if j == 0:
                lr, update, state = self.LSTM_Optimizee0(G, state_i)
                print('####################0')
            elif j == 1:
                lr, update, state = self.LSTM_Optimizee1(G, state_i)
                print('####################1')
            elif j == 2:
                lr, update, state = self.LSTM_Optimizee2(G, state_i)
                print('####################2')
            elif j == 3:
                lr, update, state = self.LSTM_Optimizee3(G, state_i)
                print('####################3')
            elif j == 4 or j == 5:
                lr, update, state = self.LSTM_Optimizee4(G, state_i)
                print('####################4')
            elif j == 6:
                lr, update, state = self.LSTM_Optimizee5(G, state_i)
                print('####################5')
            elif j == 7 or j == 8 or j == 9 or j == 10 or j==11 or j==12:
                lr, update, state = self.LSTM_Optimizee6(G, state_i)
                print('####################6')
            lr = torch.abs(lr) * s_lr

            update_R = -lr.to("cuda:2") * (update.to("cuda:2") + G.to("cuda:2"))
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


def data_preprocess():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # AutoAugment(),
        # Cutout(),
        # transforms.RandomRotation(15),
        # Cutout(),
        transforms.ToTensor(),
        normalize,
        Cutout()

    ]), download=False)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    transform=transforms.Compose([transforms.ToTensor(), normalize]),
                                    download=False)
    return train_dataset, test_dataset


def rsamo(rank, world_size):
    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print("start RSAMO___________")

    # torch.cuda.empty_cache()
    opt = config.parse_opt()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    batchsize = opt.batchsize
    lstm_layers = opt.lstm_layers
    hidden_size = opt.hidden_size
    batchsize_param = opt.batchsize_param
    observation_epoch = opt.observation_epoch
    hand_e_lr = opt.hand_e_lr
    hand_s_lr = opt.hand_s_lr
    adam_lr = opt.adam_lr
    learning_epoch = opt.epoch
    meta_e_lr = opt.e_lr
    meta_s_lr = opt.s_lr
    # print_batch_num = int(50000//batchsize//4)
    print_epoch_num = 10
    inner_epoch = opt.inner_epoch
    print(opt)
    f = open("record.txt", "w")
    str = 'e_lr: {}, s_lr: {}, adam_lr:{} ,batch_size: {}'.format(meta_e_lr, meta_s_lr, adam_lr,
                                                                  batchsize)

    train_dataset, test_dataset = data_preprocess()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batchsize,
                              shuffle=True)  # sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)  # sampler=test_sampler, shuffle=False)
    print("data teady-----")

    # local_rank = opt.local_rank
    # storch.cuda.set_device(rank)
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    # dist.init_process_group(backend='nccl')
    # init
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vgg16_BN().cuda()

    loss_function = nn.CrossEntropyLoss().cuda()
    buffersize = 1  # -----------------------------------------
    replay_buffer = RB(buffersize * batchsize_param)

    # start
    # start_time = time.time()

    def generate_params():
        # resnet = models.resnet18(pretrained=False).cuda()#.cuda().features
        # # features = torch.nn.Sequential(*list(features.children())[:-1])
        # e_params = []
        # s_params = []
        # bn_params = []
        # # init model param
        features = models.vgg16_bn(pretrained=False).features
        # features = torch.nn.Sequential(*list(features.children())[:-1])
        e_params = []
        s_params = []
        bn_params = []
        # init model param
        for m in features:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if (m.weight.shape[0]) <= m.weight.view(m.weight.shape[0], -1).shape[1]:
                    s_params.append(m.weight)
                else:
                    e_params.append(m.weight)
                m.bias.data.zero_()
                e_params.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                e_params.append(m.weight)
                e_params.append(m.bias)
                mean = nn.Parameter(torch.zeros(m.weight.shape[0]), requires_grad=False).cuda()
                var = nn.Parameter(torch.ones(m.weight.shape[0]), requires_grad=False).cuda()
                bn_params.extend([mean, var])
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                e_params.append(m.weight)
                m.bias.data.zero_()
                e_params.append(m.bias)

        # init params
        params = []
        e_params_1 = []
        s_params_1 = []
        bn_params_1 = []
        state=[]

        # conv2d
        for j, e_i in enumerate(e_params):
            e_params_i = torch.randn(e_i.shape).cuda()
            nn.init.normal_(e_params_i)  # nn.init.normal_(e_params_i, 0, math.sqrt(2./n))
            e_params_1.append(e_params_i)

        # bn
        for b_i in bn_params:
            b_i_1 = nn.Parameter(torch.zeros(b_i.shape).cuda(), requires_grad=False).cuda()
            bn_params_1.append(b_i_1)

        # classifier
        fc_weight_1 = torch.randn(10, 2048).cuda()
        nn.init.orthogonal_(fc_weight_1)
        e_params_1.append(fc_weight_1)
        fc_bias_1 = torch.randn(10).cuda()
        fc_bias_1.zero_()
        e_params_1.append(fc_bias_1)
        # fc_weight_2 = torch.randn(1024, 1024).cuda()
        # nn.init.orthogonal_(fc_weight_2)
        # e_params_1.append(fc_weight_2)
        # fc_bias_2 = torch.randn(1024).cuda()
        # fc_bias_2.zero_()
        # e_params_1.append(fc_bias_2)
        # fc_weight_3 = torch.randn(10, 1024).cuda()
        # nn.init.orthogonal_(fc_weight_3)
        # e_params_1.append(fc_weight_3)
        # fc_bias_3 = torch.randn(10).cuda()
        # fc_bias_3.zero_()
        # e_params_1.append(fc_bias_3)

        for j, s_i in enumerate(s_params):
            s_params_i = torch.randn(s_i.shape).cuda()
            nn.init.orthogonal_(s_params_i)
            s_params_1.append(s_params_i.cuda())
            state_i = None
            state.append(state_i)

        # L_before = []
        # R_before = []
        # e_length = len(e_params)


        # for j, e_i in enumerate(e_params):
        #     e_params_i = torch.randn(e_i.shape)#.cuda()
        #     nn.init.normal_(e_params_i)  # nn.init.normal_(e_params_i, 0, math.sqrt(2./n))
        #     e_params_1.append(e_params_i)

        params = {
            'e_params': e_params_1,
            's_params': s_params_1,
            'bn_params': bn_params_1,
            'state': state}
        return params



    # accuracy = []
    max_test_accu = 0.0
    min_train_loss = 999999
    min_test_loss = 999999

    def test(param, min_test_loss, max_test_accu):
        model.eval()
        correct = 0
        total = 0
        loss_total = 0
        loss_count = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs, param)
                loss = loss_function(outputs, labels)  # .cuda()
                loss_total += loss.item()
                loss_count += 1
                pred = outputs.argmax(dim=1)
                total += inputs.size(0)
                correct += torch.eq(pred, labels).sum().item()
        loss_test = round(loss_total / loss_count, 4)
        accu = round(100.0 * correct / total, 2)
        print('Accuracy of the network on the 10000 test images: %.2f %% loss:%.4f ' % (
            100.0 * correct / total, loss_test))
        str = 'Accuracy of the network: {}%%, test loss: {} \r\n'.format(accu, loss_test)
        if loss_test < min_test_loss:
            min_test_loss = loss_test
        if accu > max_test_accu:
            max_test_accu = accu
            torch.save(optimizer.state_dict(), './max_acc.pth')
        f.write(str)
        model.train()
        return min_test_loss, max_test_accu, accu, loss_test
        # accuracy.append(accu)

    # observation stage
    observation_loss = 0
    observation_loss_total = 0
    observation_optimizer = RSGD(opt)
    for i in range(observation_epoch):
        if i == 0:
            param = generate_params()
            replay_buffer.push(param)
        for e_i in param['e_params']:
            e_i.requires_grad = True
            e_i.retain_grad()
        for s_i in param['s_params']:
            s_i.requires_grad = True
            s_i.retain_grad()

        for j, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, param)
            observation_loss = loss_function(outputs, labels)  # .cuda()
            observation_loss.backward()
            new_param = observation_optimizer.step(param)
            observation_optimizer.zero_grad(new_param)
            observation_loss_total += observation_loss
            replay_buffer.push(param)

    # prepare replay_buffer
    if replay_buffer.is_full() == 0:
        num = buffersize - observation_epoch
        for i in range(num):
            param = generate_params()
            replay_buffer.push(param)
    replay_buffer.shuffle()
    print("params ready-------------")
    # print("len of replay_buffer-------------is ",replay_buffer.get_length())
    # adjust_lr
    # -----------
    #

    # learning stage
    print(rank)
    optimizer = meta_optimizer(opt=opt)
    # model = torch.nn.parallel.DistributedDataParallel(optimizer, device_ids=[rank])
    # optimizer = ddp(optimizer, device_ids=[rank])

    print('meta optimizer ready')

    adam_optimizer = torch.optim.Adam(optimizer.parameters(), lr=adam_lr)
    global_loss_num = 0
    global_loss_total = 0.0

    def adj_lr(epoch):
        e_decayed_learning_rate = meta_e_lr * opt.decay_rate
        s_decayed_learning_rate = meta_s_lr * opt.decay_rate
        # if epoch <= opt.lr_step:
        #     return max(opt.e_lr, 0.0001), max(opt.s_lr, 0.0001)
        # elif epoch <= opt.lr_step*2:
        #     return max(0.1*opt.e_lr, 0.0001), max(0.1*opt.s_lr, 0.0001)
        # else:
        #     return max(0.01*opt.e_lr, 0.0001), max(0.01*opt.s_lr, 0.0001)
        return e_decayed_learning_rate, s_decayed_learning_rate

    # losses = []
    max_train_accu = 0.0
    writer = SummaryWriter('./log')
    # train_loss = 0.0
    # test_loss = 0.0
    train_accu = 0
    test_accu = 0
    test_loss = 0.0
    # train_loss_0 = 0.0
    # test_loss_0 = 0.0
    # train
    model.train()
    start = time.clock()
    for epoch in range(learning_epoch):
        # print("start optimization")
        if epoch % opt.lr_step == 0 and epoch != 0:
            meta_e_lr = 0.1 * meta_e_lr
            meta_s_lr = 0.5 * meta_s_lr
        # train_loader.sampler.set_epoch(epoch)

        replay_buffer.shuffle()
        params = replay_buffer.sample(batchsize_param)
        # outer_loop
        # print("epoch-----",epoch,"-------len of params ",len(params))
        for i, param in enumerate(params):

            outer_loss_graph = 0.0
            for e_i in param['e_params']:
                e_i.requires_grad = True
                e_i.retain_grad()
            for s_i in param['s_params']:
                s_i.requires_grad = True
                s_i.retain_grad()
                pdb.set_trace()
                print(s_i.shape)

            # inner_loop
            inner_loss = 0
            inner_loss_total = 0.0
            correct = 0
            total = 0
            for j, (inputs, labels) in enumerate(train_loader, 1):
                if j > inner_epoch:
                    break
                if j >= inner_epoch - 1:
                    replay_buffer.push(param)
                    train_accu = round(100.0 * correct / total, 2)
                    if max_train_accu < train_accu:
                        max_train_accu = train_accu
                    params = replay_buffer.sample(batchsize_param)
                    param0 = params[0]
                    for e_i in param0['e_params']:
                        e_i.requires_grad = True
                    e_i.retain_grad()
                    for s_i in param0['s_params']:
                        s_i.requires_grad = True
                    s_i.retain_grad()

                    inputs_clone, labels_clone = inputs.clone().cuda(), labels.clone().cuda()
                    outputs = model(inputs_clone, param0)
                    validation_loss = loss_function(outputs, labels_clone).cuda()
                    validation_loss.backward()

                    output_temp = model(inputs.cuda(), param)
                    loss_temp = loss_function(output_temp, labels.cuda())
                    loss_temp.backward()

                    break_flag = optimizer.last_step(param, param0, meta_e_lr, meta_s_lr, inputs, labels, model)

                    break








                else:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs, param)
                    inner_loss = loss_function(outputs, labels).cuda()
                    # print('inner loss.....j..',j,'is ...',inner_loss.item())
                    optimizer.zero_grad(param)
                    inner_loss.backward()
                    with torch.no_grad():
                        pred = outputs.argmax(dim=1)
                        total += inputs.size(0)
                        correct += torch.eq(pred, labels).sum().item()
                        global_loss_num += 1
                    outer_loss_graph += inner_loss
                    inner_loss_total += inner_loss.item()
                    global_loss_total += inner_loss.item()
                    param = optimizer.step(param, meta_e_lr, meta_s_lr)
                # if j % print_batch_num == print_batch_num-1 :
                # if j == inner_epoch:
                #     print('inner -------- %d epoch %d innerloop loss: %.6f' %(epoch, j, inner_loss_total / inner_epoch))
                # replay_buffer.push(new_param)

            # losses.append(outer_loss_graph)

        if epoch % print_epoch_num == 0:
            time_i = time.clock()
            time_now = time_i - start
            min_test_loss, max_test_accu, test_accu, test_loss = test(param, min_test_loss, max_test_accu)
            train_loss = round(global_loss_total / global_loss_num, 4)
            print("epoch: ", epoch, "train loss: ", train_loss, "train_accu", train_accu, 'e_lr: ', meta_e_lr,
                  's_lr: ', meta_s_lr)
            str = "epoch: {}, train loss: {}, e_lr: {}, s_lr: {}, train_accu:{} %%\r\n".format(epoch, train_loss,
                                                                                               meta_e_lr, meta_s_lr,
                                                                                               train_accu)
            writer.add_scalars('accu_train_test', {'train_accu': train_accu, 'test_accu': test_accu}, epoch)
            writer.add_scalars('loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)
            writer.add_scalars('loss_time', {'train_loss': train_loss, 'test_loss': test_loss}, time_now)
            if min_train_loss > train_loss:
                min_train_loss = train_loss
            f.write(str)
            global_loss_total = 0.0
            global_loss_num = 0

    torch.save(param, './param.pth')
    str = 'min_train_loss: {}, min_test_loss: {}, max_train_accu:{} ,max_test_accu: {}'.format(min_train_loss,
                                                                                               min_test_loss,
                                                                                               max_train_accu,
                                                                                               max_test_accu)
    f.write(str)
    f.close()
    writer.close()
    torch.save(optimizer.state_dict(), './epoch-last.pth')


def main():
    world_size = 1
    rsamo(0, 1)

    # torch.multiprocessing.spawn(rsamo, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()






















