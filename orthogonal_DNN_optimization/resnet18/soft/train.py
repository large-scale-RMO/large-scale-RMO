import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config
import os
# from Model.vgg16 import vgg16
# from Model.vgg16_BN import vgg16_BN
# from Model.resnet50 import Resnet50
from Model.resnet18 import Resnet18
from optimizer.SGD import SGD
# from optimizer.meta_optimizer import meta_optimizer
from RB import RB
import numpy as np
import time
torch.cuda.empty_cache()
opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
f = open("record_srip_p.txt", "w")
def data_preprocess():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]), download=False)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(), normalize]), download=False)
    return train_dataset, test_dataset

train_dataset, test_dataset = data_preprocess()
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle= True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#init
model = Resnet18().to(device)
loss_function = nn.CrossEntropyLoss().cuda()
buffersize = 1 #-----------------------------------------
replay_buffer = RB(buffersize*batchsize_param)

#start
#start_time = time.time()

def SRIP_P(param):
    s_n = None
    for w in param['e_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            # v = Variable(torch.randn(w_tmp.shape[1], 1))
            # # v = v.cuda()
            # u = torch.matmul(w_tmp, v)
            # v_p = torch.matmul(w_tmp, u)
            # norm_v = torch.norm(v_p, 2)
            # norm_u = torch.norm(u, 2)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda()

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)
            if s_n is None:
                s_n = (torch.norm(v3, 2)) ** 2
            else:
                s_n = s_n + (torch.norm(v3, 2)) ** 2
    for w in param['s_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            # v = Variable(torch.randn(w_tmp.shape[1], 1))
            # # v = v.cuda()
            # u = torch.matmul(w_tmp, v)
            # v_p = torch.matmul(w_tmp, u)
            # norm_v = torch.norm(v_p, 2)
            # norm_u = torch.norm(u, 2)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda()

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)
            if s_n is None:
                s_n = (torch.norm(v3, 2)) ** 2
            else:
                s_n = s_n + (torch.norm(v3, 2)) ** 2
    return s_n


def SRIP(param):
    s_n = None
    for w in param['e_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w_p, w_t)
                I = Variable(torch.eye(rows, rows), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            # v = Variable(torch.randn(w_tmp.shape[1], 1))
            # # v = v.cuda()
            # u = torch.matmul(w_tmp, v)
            # v_p = torch.matmul(w_tmp, u)
            # norm_v = torch.norm(v_p, 2)
            # norm_u = torch.norm(u, 2)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda()

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)
            if s_n is None:
                s_n = (torch.norm(v3, 2)) ** 2
            else:
                s_n = s_n + (torch.norm(v3, 2)) ** 2
    for w in param['s_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w_p, w_t)
                I = Variable(torch.eye(rows, rows), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            # v = Variable(torch.randn(w_tmp.shape[1], 1))
            # # v = v.cuda()
            # u = torch.matmul(w_tmp, v)
            # v_p = torch.matmul(w_tmp, u)
            # norm_v = torch.norm(v_p, 2)
            # norm_u = torch.norm(u, 2)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda()

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)
            if s_n is None:
                s_n = (torch.norm(v3, 2)) ** 2
            else:
                s_n = s_n + (torch.norm(v3, 2)) ** 2
    return s_n


def SO(param):
    norm = None
    for w in param['e_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            I = Variable(torch.eye(cols, cols), requires_grad=True)
            I = I.cuda()
            m = torch.matmul(w_t, w_p) - I
            norm_p = (torch.norm(m, 2))  # **2
            if norm is None:
                norm = norm_p
            else:
                norm = norm + norm_p

    for w in param['s_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            I = Variable(torch.eye(cols, cols), requires_grad=True)
            I = I.cuda()
            m = torch.matmul(w_t, w_p) - I
            norm_p = (torch.norm(m, 2))  # **2
            if norm is None:
                norm = norm_p
            else:
                norm = norm + norm_p
    return norm


def DSO(param):
    norm = None
    for w in param['e_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            I_1 = Variable(torch.eye(rows, rows), requires_grad=True)   .cuda()
            I_2 = Variable(torch.eye(cols, cols), requires_grad=True)   .cuda()
            m1 = torch.matmul(w_t, w_p) - I_2
            m2 = torch.matmul(w_p, w_t) - I_1
            norm1 = (torch.norm(m1, 2))  # **2
            norm2 = (torch.norm(m2, 2))  # **2
            if norm is None:
                norm = norm1 + norm2
            else:
                norm = norm + norm1 + norm2
    for w in param['s_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            I_1 = Variable(torch.eye(rows, rows), requires_grad=True)   .cuda()
            I_2 = Variable(torch.eye(cols, cols), requires_grad=True)   .cuda()
            m1 = torch.matmul(w_t, w_p) - I_2
            m2 = torch.matmul(w_p, w_t) - I_1
            norm1 = (torch.norm(m1, 2))  # **2
            norm2 = (torch.norm(m2, 2))  # **2
            if norm is None:
                norm = norm1 + norm2
            else:
                norm = norm + norm1 + norm2
    return norm


def MC(param):
    norm = None
    for w in param['e_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w_p, w_t)
                I = Variable(torch.eye(rows, rows), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            norm_p = torch.norm(w_tmp, float('inf'))
            if norm is None:
                norm = norm_p
            else:
                norm = norm + norm_p
    for w in param['s_params']:
        if w.ndimension() < 2:
            continue
        else:
            w_p = w.view(w.shape[0], -1)
            w_t = w_p.T
            rows = w_p.shape[0]
            cols = w_p.shape[1]
            if (rows > cols):
                m = torch.matmul(w_t, w_p)
                I = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w_p, w_t)
                I = Variable(torch.eye(rows, rows), requires_grad=True)
            I = I.cuda()
            w_tmp = m - I
            norm_p = torch.norm(w_tmp, float('inf'))
            if norm is None:
                norm = norm_p
            else:
                norm = norm + norm_p
    return norm


def ad_ortho_decay(epoch):
    o_d = opt.ortho_decay
    if epoch > 12000:
        o_d = 0.0
    elif epoch > 7000:
        o_d = 1e-6 * o_d
    elif epoch > 5000:
        o_d = 1e-4 * o_d
    elif epoch > 2000:
        o_d = 1e-3 * o_d
    return o_d

def generate_params():
    resnet = models.resnet18(pretrained=False).cuda()#.features
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
                        mean = nn.Parameter(torch.zeros(n.weight.shape[0]), requires_grad=False)   .cuda()
                        var = nn.Parameter(torch.ones(n.weight.shape[0]), requires_grad=False)   .cuda()
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
                                mean = nn.Parameter(torch.zeros(i.weight.shape[0]), requires_grad=False)   .cuda()
                                var = nn.Parameter(torch.ones(i.weight.shape[0]), requires_grad=False)   .cuda()
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
    e_length = len(e_params)

    # conv2d
    for j, e_i in enumerate(e_params):
        e_params_i = torch.randn(e_i.shape)   .cuda()
        if e_params_i.ndimension() < 2:
            nn.init.normal_(e_params_i)
            e_params_1.append(e_params_i)
        else:
            nn.init.orthogonal_(e_params_i)
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
        # L_h0_i = torch.zeros(lstm_layers, s_params_i.view(s_params_i.shape[0], -1).shape[1], hidden_size).cuda()
        # L_c0_i = torch.zeros(lstm_layers, s_params_i.view(s_params_i.shape[0], -1).shape[1], hidden_size).cuda()
        # R_h0_i = torch.zeros(lstm_layers, s_params_i.shape[0], hidden_size)
        # R_c0_i = torch.zeros(lstm_layers, s_params_i.shape[0], hidden_size)
        # L_h0.append(L_h0_i)
        # L_c0.append(L_c0_i)
        # R_h0.append(R_h0_i)
        # R_c0.append(R_c0_i)

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
        'R_c0': R_c0
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
    f.write(str)
    model.train()
    return min_test_loss, max_test_accu
    # accuracy.append(accu)




# observation stage
observation_loss = 0
observation_loss_total = 0
observation_optimizer = SGD(opt)
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
optimizer = SGD(opt=opt)
# adam_optimizer = torch.optim.Adam(optimizer.parameters(), lr=adam_lr)
global_loss_num = 0
global_loss_total = 0.0

def adj_lr(epoch):
    e_decayed_learning_rate = hand_e_lr * opt.decay_rate
    s_decayed_learning_rate = hand_s_lr * opt.decay_rate
    # if epoch <= opt.lr_step:
    #     return max(opt.e_lr, 0.0001), max(opt.s_lr, 0.0001)
    # elif epoch <= opt.lr_step*2:
    #     return max(0.1*opt.e_lr, 0.0001), max(0.1*opt.s_lr, 0.0001)
    # else:
    #     return max(0.01*opt.e_lr, 0.0001), max(0.01*opt.s_lr, 0.0001)
    return e_decayed_learning_rate, s_decayed_learning_rate


# losses = []

#train
model.train()
for e_i in param['e_params']:
    e_i.requires_grad = True
    e_i.retain_grad()
for s_i in param['s_params']:
    s_i.requires_grad = True
    s_i.retain_grad()

time_iter=0
start = time.clock()
for epoch in range(learning_epoch):
    if epoch % opt.lr_step == 0 and epoch != 0:
        hand_e_lr, hand_s_lr = adj_lr(epoch)
    ortho_decay = ad_ortho_decay(epoch+1)
    for j, (inputs, labels) in enumerate(train_loader, 0):
        if j > inner_epoch:
            break
        if time_iter == 100:
            end = time.clock()
            print(end-start)
        time_iter +=1
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, param)
        loss = loss_function(outputs, labels)#.cuda()
        oloss = MC(param)
        loss = loss + ortho_decay * oloss
        global_loss_total += loss.item()
        global_loss_num += 1
        optimizer.zero_grad(param)
        loss.backward()

        param = optimizer.step(param, hand_e_lr, hand_s_lr)


str = 'min_train_loss: {}, min_test_loss: {}, max_test_accu: {}'.format(min_train_loss, min_test_loss, max_test_accu)
f.write(str)
f.close()








