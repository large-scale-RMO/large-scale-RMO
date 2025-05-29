import os
import pdb
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import config

# from Model.vgg16 import vgg1
from Model.Cutout import Cutout

# from Model.vgg16_BN import vgg16_BN
# from Model.resnet50 import Resnet50
from Model.resnet18 import Resnet18
from optimizer.RSGD import RSGD
from optimizer.meta_optimizer import meta_optimizer
from RB import RB
from tensorboardX import SummaryWriter
import numpy as np
import time
torch.cuda.empty_cache()
opt = config.parse_opt()

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
def optimizer_loss(optimizer):
    ##这个loss并不会用于训练，而是用于将参数的grad转为非none##
    optimizer_param_sum = 0

    for param in optimizer.parameters():
        optimizer_param_sum = optimizer_param_sum + torch.norm(param)
    return optimizer_param_sum
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

def data_preprocess():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data', train=True, transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        normalize    ]), download=False)
    test_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data', train=False, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize]), download=False)
    return train_dataset, test_dataset

train_dataset, test_dataset = data_preprocess()
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle= False)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
#init
model = Resnet18().to(device)
loss_function = nn.CrossEntropyLoss().cuda()
buffersize = 1 #-----------------------------------------
replay_buffer = RB(buffersize*batchsize_param)

#start
#start_time = time.time()

def generate_params():
    resnet = models.resnet18(pretrained=False).cuda()#.cuda().features
    # features = torch.nn.Sequential(*list(features.children())[:-1])
    e_params = []
    s_params = []
    bn_params = []
    # init model param
    features = torch.nn.Sequential(*list(resnet.children())[:-1])
    for m in features:
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if(m.weight.shape[0]) <= m.weight.view(m.weight.shape[0], -1).shape[1]:
                s_params.append(m.weight)
            else:
                e_params.append(m.weight)
            # m.bias.data.zero_()
            # e_params.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            e_params.append(m.weight)
            e_params.append(m.bias)
            mean = nn.Parameter(torch.zeros(m.weight.shape[0]),requires_grad=False).cuda()
            var = nn.Parameter(torch.ones(m.weight.shape[0]),requires_grad=False).cuda()
            bn_params.extend([mean, var])
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            e_params.append(m.weight)
            m.bias.data.zero_()
            e_params.append(m.bias)
        elif isinstance(m, nn.Sequential):
            for k in m:
                ks = torch.nn.Sequential(*list(k.children()))
                for n in ks:
                    if isinstance(n, nn.Conv2d):
                        nn.init.orthogonal_(n.weight.data)
                        if (n.weight.shape[0]) <= n.weight.view(n.weight.shape[0], -1).shape[1]:
                            s_params.append(n.weight)
                        else:
                            e_params.append(n.weight)
                        # m.bias.data.zero_()
                        # e_params.append(m.bias)
                    elif isinstance(n, nn.BatchNorm2d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()
                        e_params.append(n.weight)
                        e_params.append(n.bias)
                        mean = nn.Parameter(torch.zeros(n.weight.shape[0]), requires_grad=False).cuda()
                        var = nn.Parameter(torch.ones(n.weight.shape[0]), requires_grad=False).cuda()
                        bn_params.extend([mean, var])
                    elif isinstance(n, nn.Sequential):
                        for i in n:
                            if isinstance(i, nn.Conv2d):
                                nn.init.orthogonal_(i.weight.data)
                                if (i.weight.shape[0]) <= i.weight.view(i.weight.shape[0], -1).shape[1]:
                                    s_params.append(i.weight)
                                else:
                                    e_params.append(i.weight)
                                # m.bias.data.zero_()
                                # e_params.append(m.bias)
                            elif isinstance(i, nn.BatchNorm2d):
                                i.weight.data.fill_(1)
                                i.bias.data.zero_()
                                e_params.append(i.weight)
                                e_params.append(i.bias)
                                mean = nn.Parameter(torch.zeros(i.weight.shape[0]), requires_grad=False).cuda()
                                var = nn.Parameter(torch.ones(i.weight.shape[0]), requires_grad=False).cuda()
                                bn_params.extend([mean, var])


    # init params
    params = []
    e_params_1 = []
    s_params_1 = []
    bn_params_1 = []
    L_h0 = []
    L_c0 = []
    R_h0 = []
    R_c0 = []
    L_before = []
    R_before = []
    e_length = len(e_params)

    # conv2d
    for j, e_i in enumerate(e_params):

        e_params_i = torch.randn(e_i.shape).cuda()
        nn.init.normal_(e_params_i) # nn.init.normal_(e_params_i, 0, math.sqrt(2./n))
        e_params_1.append(e_params_i)

    #bn
    for b_i in bn_params:
        b_i_1 = nn.Parameter(torch.zeros(b_i.shape).cuda(), requires_grad=False).cuda()
        bn_params_1.append(b_i_1)

    # classifier
    fc_weight_1 = torch.randn(10, 512).cuda()
    nn.init.orthogonal_(fc_weight_1)
    e_params_1.append(fc_weight_1)
    fc_bias_1 = torch.randn(10).cuda()
    fc_bias_1.zero_()
    e_params_1.append(fc_bias_1)

    for j, s_i in enumerate(s_params):
        s_params_i = torch.randn(s_i.shape).cuda()
        nn.init.orthogonal_(s_params_i)
        s_params_1.append(s_params_i)
        L_h0_i = torch.zeros(lstm_layers, s_params_i.view(s_params_i.shape[0], -1).shape[1], hidden_size).cuda()
        L_c0_i = torch.zeros(lstm_layers, s_params_i.view(s_params_i.shape[0], -1).shape[1], hidden_size).cuda()
        R_h0_i = torch.zeros(lstm_layers, s_params_i.shape[0], hidden_size)
        R_c0_i = torch.zeros(lstm_layers, s_params_i.shape[0], hidden_size)
        L_h0.append(L_h0_i)
        L_c0.append(L_c0_i)
        R_h0.append(R_h0_i)
        R_c0.append(R_c0_i)
        M = s_i.data.view(s_i.shape[0], -1).T
        n = M.shape[0]
        r = M.shape[1]
        L_0 = torch.zeros(n, n)
        R_0 = torch.zeros(r, r)
        L_before.append(L_0.cuda())  # .cuda())
        R_before.append(R_0.cuda())  # .cuda())

    # print("len of e",len(e_params_1))
    # print("len of s", len(s_params_1))
    # print("len of bn", len(bn_params_1))

    params = {
        'e_params': e_params_1,
        's_params': s_params_1,
        'bn_params': bn_params_1,
        'L_h0': L_h0,
        'L_c0': L_c0,
        'R_h0': R_h0,
        'R_c0': R_c0,
        'L_before': L_before,
        'R_before': R_before
    }
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, param)
            loss = loss_function(outputs, labels).to(device)
            loss_total += loss.item()
            loss_count += 1
            pred = outputs.argmax(dim=1)
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()
    loss_test = round(loss_total / loss_count , 4)
    accu = round(100.0 * correct / total, 2)
    print('Accuracy of the network on the 10000 test images: %.2f %% loss:%.4f ' % (100.0 * correct / total, loss_test))
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
    for l_h0_i in param['L_h0']:
        l_h0_i.requires_grad = True
        l_h0_i.retain_grad()
    for l_c0_i in param['L_c0']:
        l_c0_i.requires_grad = True
        l_c0_i.retain_grad()
    for r_h0_i in param['R_h0']:
        r_h0_i.requires_grad = True
        r_h0_i.retain_grad()
    for r_c0_i in param['R_c0']:
        r_c0_i.requires_grad = True
        r_c0_i.retain_grad()

    for j, (inputs, labels) in enumerate(train_loader):
        # inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, param)
        observation_loss = loss_function(outputs, labels)#.to(device)
        observation_loss.backward()
        new_param = observation_optimizer.step(param)
        observation_optimizer.zero_grad(new_param)
        observation_loss_total += observation_loss
        replay_buffer.push(param)

    # test
    #--------------------
    #

# prepare replay_buffer
if replay_buffer.is_full() == 0:
    num = buffersize - observation_epoch
    for i in range(num):
        param = generate_params()
        replay_buffer.push(param)
replay_buffer.shuffle()

# print("len of replay_buffer-------------is ",replay_buffer.get_length())
#adjust_lr
#-----------
#

# learning stage
optimizer = meta_optimizer(opt=opt)
#para = sum([np.prod(list(p.size())) for p in optimizer.parameters()])
#print('Model {} : params: {:4f}'.format(optimizer._get_name(), para * 4 / 1024))
#os._exit()
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
#train
model.train()
start = time.clock()
for epoch in range(learning_epoch):
    if epoch % opt.lr_step == 0 and epoch != 0:
        meta_e_lr = 0.1 * meta_e_lr
    if epoch == 6000:
        meta_s_lr = 0.005
    if epoch == 9000:
        meta_s_lr = 0.002
    if epoch == 11000:
        meta_s_lr = 0.0008
    if epoch == 12000:
        meta_s_lr = 0.0003
    if epoch == 13000:
        meta_s_lr = 0.00001

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
        for l_h0_i in param['L_h0']:
            l_h0_i.requires_grad = True
            l_h0_i.retain_grad()
        for l_c0_i in param['L_c0']:
            l_c0_i.requires_grad = True
            l_c0_i.retain_grad()
        for r_h0_i in param['R_h0']:
            r_h0_i.requires_grad = True
            r_h0_i.retain_grad()
        for r_c0_i in param['R_c0']:
            r_c0_i.requires_grad = True
            r_c0_i.retain_grad()

        # inner_loop
        inner_loss = 0
        inner_loss_total = 0.0
        correct = 0
        total = 0
        NAN_error = False
        break_flag = False
        start = time.time()
        for j, (inputs, labels) in enumerate(train_loader, 1):
            if j > inner_epoch:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, param)
            inner_loss = loss_function(outputs, labels).to(device)
            # print('inner loss.....j..',j,'is ...',inner_loss.item())
            optimizer.zero_grad(param)
            inner_loss.backward()
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                total += inputs.size(0)
                correct += torch.eq(pred, labels).sum().item()
                global_loss_num += 1
            # outer_loss_graph += inner_loss
            inner_loss_total += inner_loss.item()
            global_loss_total += inner_loss.item()
            # pdb.set_trace()
            param = optimizer.step(param, meta_e_lr, meta_s_lr)
            if j == inner_epoch-1:
                for e_i in param['e_params']:
                    e_i.detach()
                    e_i.requires_grad = True
                    e_i.retain_grad()
                for s_i in param['s_params']:
                    s_i.detach()
                    s_i.requires_grad = True
                    s_i.retain_grad()
                for l_h0_i in param['L_h0']:
                    l_h0_i.detach()
                    l_h0_i.requires_grad = True
                    l_h0_i.retain_grad()
                for l_c0_i in param['L_c0']:
                    l_c0_i.detach()
                    l_c0_i.requires_grad = True
                    l_c0_i.retain_grad()
                for r_h0_i in param['R_h0']:
                    r_h0_i.detach()
                    r_h0_i.requires_grad = True
                    r_h0_i.retain_grad()
                for r_c0_i in param['R_c0']:
                    r_c0_i.detach()
                    r_c0_i.requires_grad = True
                    r_c0_i.retain_grad()
                outputs_temp = model(inputs, param)
                inner_loss_temp = loss_function(outputs_temp, labels).to(device)
                inner_loss_temp.backward()
                e_lr = opt.e_lr
                s_lr = opt.s_lr


                new_e_params = []
                # euc_update
                for e_i in param['e_params']:
                    if e_i.grad is None:
                        continue
                    e_i_g = e_i.grad
                    if opt.weight_decay != 0:
                        e_i_g = e_i_g.add(e_i.data, alpha=opt.weight_decay)
                    new_e_i = e_i.data - e_lr * e_i_g
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

                for j, s_i in enumerate(param['s_params']):
                    if s_i.grad is None:
                        continue

                    s_i_shape = s_i.shape
                    M_grad = s_i.grad.view(s_i.shape[0], -1).T
                    M_grad_p = s_lr * M_grad
                    M = s_i.data.view(s_i.shape[0], -1).T
                    M.requires_grad = True
                    r = M.shape[0]
                    c = M.shape[1]
                    Mshape = M.shape
                    G = othogonal_projection(M, M_grad_p)
                    GGt = G.mm(G.T)
                    L_input = optimizer.LogAndSign_Preprocess_Gradient(GGt)
                    GtG = G.T.mm(G)
                    R_input = optimizer.LogAndSign_Preprocess_Gradient(GtG)

                    L_h0_i = param['L_h0'][j].cuda()
                    L_c0_i = param['L_c0'][j].cuda()
                    R_h0_i = param['R_h0'][j].cuda()
                    R_c0_i = param['R_c0'][j].cuda()
                    L_before = param['L_before'][j].cuda()
                    R_before = param['R_before'][j].cuda()

                    l_output, (hn_l, cn_l) = optimizer.lstm(L_input, (L_h0_i, L_c0_i))
                    r_output, (hn_r, cn_r) = optimizer.lstm(R_input, (R_h0_i, R_c0_i))
                    l_ = optimizer.L_linear(
                        l_output.squeeze()) * optimizer.output_scale  # 因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上
                    l_ = l_ - l_.mean() + 1.0
                    L = l_.squeeze().diag()
                    # L = torch.diag_embed(L)
                    r_ = optimizer.R_linear(r_output.squeeze()) * optimizer.output_scale
                    r_ = r_ - r_.mean() + 1.0
                    R = r_.squeeze().diag()
                    # R = torch.diag_embed(R)
                    new_L_h0.append(hn_l.detach())
                    new_L_c0.append(cn_l.detach())
                    new_R_h0.append(hn_r.detach())
                    new_R_c0.append(cn_r.detach())
                    # L = torch.pow(L, -0.25).diag()
                    # R = torch.pow(R, -0.25).diag()
                    L = torch.max(L, L_before)
                    R = torch.max(R, R_before)

                    new_L_before.append(L.detach())
                    new_R_before.append(R.detach())

                    P = L.mm(G).mm(R)
                    # P = -s_lr*new_Gt
                    update_R = othogonal_projection(M, P)
                    M_end = retraction(M, -update_R).T

                    vector = G.detach().clone().T
                    Jacobi_vector = vector
                    vector_temp_list = []

                    for i in range(opt.Neumann_series):
                        vector_temp = torch.autograd.grad(M_end, update_R, grad_outputs=vector, retain_graph=True)[0]

                        # vector_temp_list.append(vector_temp.detach().cpu().numpy())

                        vector_temp = torch.autograd.grad(update_R, M, grad_outputs=vector_temp, retain_graph=True)[0]

                        vector_temp2 = torch.autograd.grad(M_end, M, grad_outputs=vector, retain_graph=True)[0]

                        vector = opt.Neumann_alpha * (vector_temp + vector_temp2).T

                        Jacobi_vector = Jacobi_vector + vector
                    Jacobi_vector_sum = torch.sum(Jacobi_vector).detach().cpu().numpy().tolist()

                    if np.isnan(Jacobi_vector_sum):
                        vector_temp_len = len(vector_temp_list)

                        for len_i in vector_temp_list:
                            print(len_i)

                        print('ERROR NAN')

                        time.sleep(5)

                        break_flag = True
                        break
                    vector_M_update = torch.autograd.grad(M_end, update_R, grad_outputs=Jacobi_vector,
                                                          retain_graph=True)

                    if j ==0:
                        update_R_optimizee = torch.autograd.grad(update_R, optimizer.parameters(),
                                                             grad_outputs=vector_M_update, retain_graph=True)
                    else:
                        update_R_optimizee = update_R_optimizee + torch.autograd.grad(update_R, optimizer.parameters(),
                                                             grad_outputs=vector_M_update, retain_graph=True)

                check = 0
                for p in optimizer.parameters():
                    check = check + 1 if type(p.grad) == type(None) else check
                if check > 0:
                    back_loss = optimizer_loss(optimizer)

                    back_loss.backward()
                for i, p in enumerate(optimizer.parameters()):
                    p.grad = update_R_optimizee[i]

                params = list(optimizer.named_parameters())
                (name, network_weight) = params[0]
                network_weight_copy = network_weight.clone()

                network_weight_shape = network_weight_copy.shape
                network_weight_length = len(network_weight_shape)
                network_weight_size = 1
                for l in range(network_weight_length):
                    network_weight_size = network_weight_size * network_weight_shape[l]

                network_weight_grad_copy = -update_R_optimizee[0]
                grad_mean = torch.sum(torch.norm(network_weight_grad_copy, p=1,
                                                 dim=(0))).detach().cpu().numpy().tolist() / network_weight_size
                print('grad_mean:{}'.format(grad_mean))

                adam_optimizer.step()
                end = time.time()
                #print('time:', end - start)
                #pdb.set_trace()
                params = list(optimizer.named_parameters())

                (name, network_weight_after) = params[0]
                contrast = network_weight_after - network_weight_copy
                loss_con = torch.sum(
                    torch.norm(contrast, p=1, dim=(0))).detach().cpu().numpy().tolist() / network_weight_size

                print('contrast:{}'.format(loss_con))
                # optimizer_iteration = optimizer_iteration + 1
                break_flag = True

                break

            # if j % print_batch_num == print_batch_num-1 :
            # if j == inner_epoch:
            #     print('inner -------- %d epoch %d innerloop loss: %.6f' %(epoch, j, inner_loss_total / inner_epoch))
            # replay_buffer.push(new_param)

        train_accu = round(100.0 * correct / total, 2)
        if max_train_accu < train_accu:
            max_train_accu = train_accu

        # adam_optimizer.zero_grad()
        # outer_loss_graph.backward()
        # adam_optimizer.step()

        replay_buffer.push(param)
        # losses.append(outer_loss_graph)

    if epoch % print_epoch_num == 0:
        time_i = time.clock()
        time_now = time_i - start
        min_test_loss, max_test_accu, test_accu, test_loss = test(param, min_test_loss, max_test_accu)
        train_loss = round(global_loss_total / global_loss_num, 4)
        print("epoch: ", epoch, "train loss: ", train_loss, "train_accu", train_accu, 'e_lr: ', meta_e_lr,
              's_lr: ', meta_s_lr)
        str = "epoch: {}, train loss: {}, e_lr: {}, s_lr: {}, train_accu:{} %%\r\n".format(epoch, train_loss, meta_e_lr,
                                                                                           meta_s_lr, train_accu)
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
torch.save(optimizer.state_dict(), './epoch-last.pth')
writer.close()
