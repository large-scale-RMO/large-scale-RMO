import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'

os.environ['MASTER_PORT']='6007'
import torch
torch.set_num_threads(1)
#torch.cuda.set_device('cuda:1') 
print(torch.cuda.current_device())
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import config
import pdb
# from Model.vgg16 import vgg1
from Model.Cutout import Cutout

# from Model.vgg16_BN import vgg16_BN
from Model.resnet50 import Resnet50
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

def data_preprocess():
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

    train_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data/', train=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    Cutout()
                                                    
        
    ]), download=False)
    test_dataset = datasets.CIFAR10(root='/home/smbu/mcislab/ypl/rsamo/cifar10/data/', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]), download=False)
    return train_dataset, test_dataset

train_dataset, test_dataset = data_preprocess()
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle= False)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device)
#init
model = Resnet50().cuda()#.to(device)
#pdb.set_trace()
loss_function = nn.CrossEntropyLoss()#.to("cuda:2")
buffersize = 1 #-----------------------------------------
replay_buffer = RB(buffersize*batchsize_param)

#start
#start_time = time.time()

def generate_params():
    resnet = models.resnet50(pretrained=False).cuda()#.features
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
                s_params.append(m.weight)
            # m.bias.data.zero_()
            # e_params.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            e_params.append(m.weight)
            e_params.append(m.bias)
            mean = nn.Parameter(torch.zeros(m.weight.shape[0]),requires_grad=False)#.cuda()
            var = nn.Parameter(torch.ones(m.weight.shape[0]),requires_grad=False)#.cuda()
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
                            s_params.append(n.weight)
                        # m.bias.data.zero_()
                        # e_params.append(m.bias)
                    elif isinstance(n, nn.BatchNorm2d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()
                        e_params.append(n.weight)
                        e_params.append(n.bias)
                        mean = nn.Parameter(torch.zeros(n.weight.shape[0]), requires_grad=False)  # .cuda()
                        var = nn.Parameter(torch.ones(n.weight.shape[0]), requires_grad=False)  # .cuda()
                        bn_params.extend([mean, var])
                    elif isinstance(n, nn.Sequential):
                        for i in n:
                            if isinstance(i, nn.Conv2d):
                                nn.init.orthogonal_(i.weight.data)
                                if (i.weight.shape[0]) <= i.weight.view(i.weight.shape[0], -1).shape[1]:
                                    s_params.append(i.weight)
                                else:
                                    s_params.append(i.weight)
                                # m.bias.data.zero_()
                                # e_params.append(m.bias)
                            elif isinstance(i, nn.BatchNorm2d):
                                i.weight.data.fill_(1)
                                i.bias.data.zero_()
                                e_params.append(i.weight)
                                e_params.append(i.bias)
                                mean = nn.Parameter(torch.zeros(i.weight.shape[0]), requires_grad=False)  # .cuda()
                                var = nn.Parameter(torch.ones(i.weight.shape[0]), requires_grad=False)  # .cuda()
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
    fc_weight_1 = torch.randn(10, 2048).cuda()
    nn.init.orthogonal_(fc_weight_1)
    e_params_1.append(fc_weight_1)
    fc_bias_1 = torch.randn(10).cuda()
    fc_bias_1.zero_()
    e_params_1.append(fc_bias_1)

    for j, s_i in enumerate(s_params):
        if j==0:
            s_params_i = torch.randn((64,3,3,3)).cuda()
        else:
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
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, param)
            loss = loss_function(outputs, labels)
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
optimizer = meta_optimizer(opt=opt)#.to("cuda:2")
#para = sum([np.prod(list(p.size())) for p in optimizer.parameters()])
#print('Model {} : params: {:4f}'.format(optimizer._get_name(), para * 4 / 1024))
#os._exit()
adam_optimizer = torch.optim.Adam(optimizer.parameters(), lr=adam_lr)#.to("cuda:2")
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

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, learning_epoch)
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
    if epoch % opt.lr_step == 0 and epoch != 0 :
        meta_e_lr=0.1*meta_e_lr
    if epoch== 3000:
        meta_s_lr= 0.5*meta_s_lr
    if epoch== 11000:
        meta_s_lr= 0.5*meta_s_lr
    if epoch== 15000:
        meta_s_lr= 0.2*meta_s_lr


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
        for j, (inputs, labels) in enumerate(train_loader, 1):
            if j > inner_epoch:
                break
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, param)
            inner_loss = loss_function(outputs, labels)
            # print('inner loss.....j..',j,'is ...',inner_loss.item())

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
            #if j % print_batch_num == print_batch_num-1 :
            # if j == inner_epoch:
            #     print('inner -------- %d epoch %d innerloop loss: %.6f' %(epoch, j, inner_loss_total / inner_epoch))
            # replay_buffer.push(new_param)

        train_accu = round(100.0 * correct / total, 2)
        if max_train_accu < train_accu:
            max_train_accu = train_accu
        adam_optimizer.zero_grad()
        outer_loss_graph.backward()
        adam_optimizer.step()
        if opt.sche==1:
            scheduler.step()
        replay_buffer.push(param)
        # losses.append(outer_loss_graph)


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
        writer.add_scalars('loss_time', {'train_loss': train_loss, 'test_loss': test_loss},time_now)
        if min_train_loss > train_loss:
            min_train_loss = train_loss
        f.write(str)
        global_loss_total = 0.0
        global_loss_num = 0
print("min_train_loss:", min_train_loss," min_test_loss:", min_test_loss, "max_test_accu:",max_test_accu)
str = 'min_train_loss: {}, min_test_loss: {}, max_train_accu:{} ,max_test_accu: {}'.format(min_train_loss, min_test_loss,max_train_accu,
                                                                            max_test_accu)
f.write(str)
f.close()
writer.close()








