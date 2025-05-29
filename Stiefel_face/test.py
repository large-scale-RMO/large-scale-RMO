import torch
import torch.nn as nn
import math
from DataSet.YaleB import YaleB
from rsamo import meta_optimizer

import config
import numpy as np
# from utils import FastRandomIdentitySampler
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
opt = config.parse_opt()
def f_ac(inputs,M):

    X=torch.matmul(M.permute(0,2,1),inputs)

    return X

# retraction=Retraction(1)

def nll_loss(data, label):
    label = label.item()
    n = data.shape[0]
    L = 0
    for i in range(n):
        L = L + torch.abs(data[i][label[i]])
    return L


def f(inputs, target, M):
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    X = torch.matmul(inputs.squeeze(), M.squeeze())
    target = target.squeeze()
    L = loss_function(X, target)

    return L


def retraction(inputs, grad, lr):
    P = -lr * grad
    PV = inputs + P
    temp_mul = torch.transpose(PV, 0, 1).mm(PV)
    e_0, v_0 = torch.symeig(temp_mul, eigenvectors=True)
    e_0 = abs(e_0) + 1e-6

    e_0 = e_0.pow(-0.5)
    temp1 = v_0.mm(torch.diag(e_0)).mm(torch.transpose(v_0, 0, 1))
    temp = PV.mm(temp1)

    return temp


def par_projection(M, update):
    M_temp = torch.matmul(torch.transpose(M, 0, 1), update)
    M_temp = 0.5 * (M_temp + torch.transpose(M_temp, 0, 1))
    M_update = update - torch.matmul(M, M_temp)

    return M_update


batchsize_data = 64
train_mnist = YaleB(opt.datapath, train=False)
train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=True, num_workers=0)

train_loader_all = torch.utils.data.DataLoader(
        train_mnist, batch_size=1920,shuffle=True, drop_last=True, num_workers=0)

train_test_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=514,shuffle=True, drop_last=True, num_workers=0)
print(batchsize_data)

LSTM_Optimizee = meta_optimizer(opt).cuda()

State_optimizer = torch.load(
    'STATE3999_0.1_Decay40000_Observe2000_Epochs100000_Optimizee_Train_Steps5000_train_steps10_hand_optimizer_lr1e-05nopretrain_newlr_meanvar_devide2.pth')

LSTM_Optimizee.load_state_dict(State_optimizer, strict=False)

Epoches = 500
print(Epoches)
DIM = 1024
outputDIM = 38
batchsize_para = 1
X = []
Y = []
X_ours = []
Y_ours = []
X_csgd = []
Y_csgd = []
iterations = 0
all_iter = 0
count = 20
N = 1900
device = torch.device("cuda:0")
data1 = np.load('data/YaleB_train_3232.npy')
data2 = torch.from_numpy(data1)
data2 = data2.float()
Data = data2.to(device)

LABELS = np.load('data/YaleB_train_gnd.npy')
LABELS = torch.from_numpy(LABELS)
LABELS = LABELS.long()
LABELS = LABELS - 1

LABELS = LABELS.to(torch.device("cuda:0"))
LABELS = LABELS.squeeze()
LABELS = LABELS.view(batchsize_para, LABELS.shape[0] // batchsize_para)
Data = Data.view(batchsize_para, Data.shape[0] // batchsize_para, -1)

learning_rate = 1e-6

# you can change the initial weigth by torch.randn() and nn.init.orthogonal_()
# w0=np.load('w1.npy')
# w1=torch.from_numpy(w0)
# w1=w1.float()
# w1=w1.to(device)

# theta0=torch.empty(DIM,outputDIM,dtype=torch.float,device=device,requires_grad=True)
# theta0=w1

M = torch.randn(opt.batchsize_para, DIM, outputDIM).cuda()
M.requires_grad = True

for i in range(opt.batchsize_para):
    nn.init.orthogonal_(M[i])

state = (torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
         torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
         torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
         torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
         torch.zeros(batchsize_para, outputDIM, outputDIM).cuda(),
         torch.zeros(batchsize_para, DIM, DIM).cuda(),
         )

for i in range(Epoches):
    it = 0
    for j, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs = inputs.view(batchsize_para, inputs.shape[0] // batchsize_para, -1)
        labels = labels.view(batchsize_para, labels.shape[0] // batchsize_para)

        loss = f(inputs, labels, M)
        loss.backward()

        train_svd, M, next_state0, next_state1, next_state2, next_state3, next_state4, next_state5 = LSTM_Optimizee.step(
            opt.s_lr, M, M.grad, state)
        state = (next_state0, next_state1, next_state2, next_state3, next_state4, next_state5)
        if train_svd == True:
            continue

        loss_test = f(Data, LABELS, M)
        correct = 0
        print('all_iter:{},train_loss:{}'.format(all_iter, loss_test.item() / N))
        if all_iter%10 == 0:
            correct = 0
            total = 0
            loss_total = 0
            loss_count = 0
            for j, data in enumerate(train_test_loader, 0):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()

                inputs = inputs.view(batchsize_para, inputs.shape[0] // batchsize_para, -1)
                # labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para,-1)
                inputs = inputs.permute(0, 2, 1)
                labels = labels - 1

                output = f_ac(inputs, M)
                output = output.permute(0, 2, 1)
                output = torch.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))
                # loss = criterion(output, labels)
                # loss_total += loss.item()
                loss_count += 1
                pred = output.argmax(dim=1).unsqueeze(1)
                correct += torch.eq(pred, labels).sum().item()
                total += labels.size(0)
            accu = round(100.0 * correct / total, 2)
            loss_test = round(loss_total / loss_count, 4)
            print('-------------------------accu',accu,'correct',correct, 'total',total)

        

        all_iter = all_iter + 1

    if i >= Epoches:
        print('END')
        break





