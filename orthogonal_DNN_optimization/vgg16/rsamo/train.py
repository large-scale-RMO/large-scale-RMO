import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import config
import os
import pdb

from Model.vgg16_BN import vgg16_BN
from Model.Cutout import Cutout

from optimizer.RSGD import RSGD
from optimizer.meta_optimizer import meta_optimizer
from RB import RB
from tensorboardX import SummaryWriter
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
f = open("record.txt", "w")
str = 'e_lr: {}, s_lr: {}, adam_lr:{} ,batch_size: {}'.format(meta_e_lr, meta_s_lr, adam_lr,
                                                                            batchsize)
def data_preprocess():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data', train=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #AutoAugment(),
        #Cutout(),
        #transforms.RandomRotation(15),
        #Cutout(),
        transforms.ToTensor(),
        normalize,
        Cutout()

    ]), download=False)
    test_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]), download=False)
    return train_dataset, test_dataset

train_dataset, test_dataset = data_preprocess()
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle= False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#init
model = vgg16_BN().to(device)
loss_function = nn.CrossEntropyLoss().cuda()
buffersize = 1 #-----------------------------------------
replay_buffer = RB(buffersize*batchsize_param)

#start
#start_time = time.time()

def generate_params():
    # resnet = models.resnet18(pretrained=False).cuda()#.cuda().features
    # # features = torch.nn.Sequential(*list(features.children())[:-1])
    # e_params = []
    # s_params = []
    # bn_params = []
    # # init model param
    features = models.vgg16_bn(pretrained=False).cuda().features
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
    fc_weight_1 = torch.randn(10, 2048).cuda()
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
            loss = loss_function(outputs, labels)#.to(device)
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
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, param)
        observation_loss = loss_function(outputs, labels)#.to(device)
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



# learning stage
optimizer = meta_optimizer(opt=opt)
adam_optimizer = torch.optim.Adam(optimizer.parameters(), lr=adam_lr)
global_loss_num = 0
global_loss_total = 0.0

def adj_lr(epoch):
    e_decayed_learning_rate = meta_e_lr * opt.decay_rate
    s_decayed_learning_rate = meta_s_lr * opt.decay_rate

    return e_decayed_learning_rate, s_decayed_learning_rate



max_train_accu = 0.0
writer = SummaryWriter('./log')

train_accu = 0
test_accu = 0
test_loss = 0.0

model.train()
start = time.clock()
for epoch in range(learning_epoch):
    # meta_s_lr为黎曼梯度前乘的缩放因子
    if epoch % opt.lr_step == 0 and epoch != 0 :
        meta_e_lr=0.1*meta_e_lr
    if epoch % opt.lr_step == 0 and epoch != 0 :
        meta_s_lr=0.1*meta_s_lr

    #手动调参
    # if epoch== 3000:
    #     meta_s_lr= 0.05
    # if epoch== 6000:
    #     meta_s_lr= 0.01
    # if epoch== 9000:
    #     meta_s_lr= 0.005
    # if epoch== 11000:
    #     meta_s_lr= 0.001


    replay_buffer.shuffle()
    params = replay_buffer.sample(batchsize_param)

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
        start = time.time()
        for j, (inputs, labels) in enumerate(train_loader, 1):
            if j > inner_epoch:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, param)
            inner_loss = loss_function(outputs, labels).to(device)
            # print('inner loss.....j..',j,'is ...',inner_loss.item())
            optimizer.zero_grad(param)
            inner_loss.backward(retain_graph=True)
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                total += inputs.size(0)
                correct += torch.eq(pred, labels).sum().item()
                global_loss_num += 1
            outer_loss_graph += inner_loss
            inner_loss_total += inner_loss.item()
            global_loss_total += inner_loss.item()
            param = optimizer.step(param, meta_e_lr, meta_s_lr)


        train_accu = round(100.0 * correct / total, 2)
        if max_train_accu < train_accu:
            max_train_accu = train_accu
            
        adam_optimizer.zero_grad()
        outer_loss_graph.backward()
        adam_optimizer.step()
        end = time.time()
        print('time:', end - start)


        replay_buffer.push(param)


    if epoch % print_epoch_num == 0 :
        time_i = time.clock()
        time_now = time_i-start
        min_test_loss, max_test_accu,test_accu, test_loss = test(param, min_test_loss, max_test_accu)
        train_loss = round(global_loss_total / global_loss_num, 4)
        print("epoch: ", epoch, "train loss: ", train_loss, "train_accu", train_accu, 'e_lr: ', meta_e_lr,
              's_lr: ', meta_s_lr)
        str = "epoch: {}, train loss: {}, e_lr: {}, s_lr: {}, train_accu:{} %%\r\n".format(epoch,train_loss, meta_e_lr, meta_s_lr,train_accu)
        writer.add_scalars('accu_train_test', {'train_accu':train_accu, 'test_accu': test_accu}, epoch)
        writer.add_scalars('loss',{'train_loss': train_loss, 'test_loss': test_loss}, epoch)
        writer.add_scalars('loss_time', {'train_loss': train_loss, 'test_loss': test_loss}, time_now)
        if min_train_loss > train_loss:
            min_train_loss = train_loss
        f.write(str)
        global_loss_total = 0.0
        global_loss_num = 0


torch.save(param, './param.pth')

str = 'min_train_loss: {}, min_test_loss: {}, max_train_accu:{} ,max_test_accu: {}'.format(min_train_loss, min_test_loss,max_train_accu,
                                                                            max_test_accu)
f.write(str)
f.close()
torch.save(optimizer.state_dict(), './epoch-last.pth')
writer.close()









