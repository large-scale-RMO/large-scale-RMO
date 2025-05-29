import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random
import numpy as np
from torch import autograd

from losses.LOSS import ContrastiveLoss
from ReplyBuffer import ReplayBuffer
from retraction import Retraction
import pdb
# retraction = Retraction(1)
grad_list = []
def retraction_(M, P):
    A = M + P
    Q, R = A.qr()
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out


criterion = nn.CrossEntropyLoss()
def f_ac(inputs, M):
    X = torch.matmul(inputs.squeeze(), M.squeeze())

    return X

def othogonal_projection(M, M_grad):
    A = M.T.mm(M_grad)

    new = M_grad - M.mm(A)
    return new

retraction=Retraction(1)
def print_grad(grad):
    grad_list.append(grad)


def nll_loss(data, label):
    n = data.shape[0]
    L = 0
    for i in range(n):
        L = L + torch.abs(data[i][label[i]])
    return L


def compute_loss(M, data, batchsize_para):
    inputs, labels = data
    inputs_clone = inputs.clone().cuda()
    labels_clone = labels.clone().cuda()

    inputs_clone = inputs_clone.view(batchsize_para, inputs_clone.shape[0] // batchsize_para, -1)
    labels_clone = labels_clone.view(batchsize_para, labels_clone.shape[0] // batchsize_para)

    loss = f(inputs_clone, labels_clone,M)
    # output = output.permute(0, 2, 1)

    return loss

def f(inputs, target, M):
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
    # M=M.permute(0,2,1)
    # print('inputs:{}'.format(inputs.shape))
    # print('M:{}'.format(M.shape))
    X = torch.matmul(inputs, M)
    n = M.shape[0]
    L = 0
    for i in range(n):
        X_temp = torch.squeeze(X[i], 0)
        # y_pred=torch.nn.functional.log_softmax(X_temp)
        # print(y_pred)
        target_temp = torch.squeeze(target[i], 0)
        # print(X_temp)
        # print(target_temp)
        # L=L+nll_loss(y_pred,target_temp)
        L = L + loss_function(X_temp, target_temp)

    return L
def optimizee_loss(optimizer):

    optimizer_param_sum = 0

    for param in optimizer.parameters():
        optimizer_param_sum = optimizer_param_sum + torch.norm(param)
    return optimizer_param_sum

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def learning_rate_multi(lr, update):
    n = lr.shape[0]
    for i in range(n):
        learning_rate = lr[i]
        update_temp = update[i] * learning_rate
        update[i] = update_temp
        # print('---------temp-----------',update_temp.shape)
    # print(update)
    # print('---------------',update.shape)
    return update


def Learning_to_learn_global_training(opt, hand_optimizee, optimizee, train_loader, train_test_loader):
    DIM = opt.DIM
    outputDIM = opt.outputDIM
    batchsize_para = opt.batchsize_para
    Observe = opt.Observe
    Epochs = opt.Epochs
    Optimizee_Train_Steps = opt.Optimizee_Train_Steps
    optimizer_lr = opt.optimizer_lr
    Decay = opt.Decay
    s_lr = opt.s_lr
    Decay_rate = opt.Decay_rate
    Imcrement = opt.Imcrement
    Sample_number = opt.Sample_number
    X = []
    Y = []
    network_contrast = []
    network_gradient = []

    M_after_gradient = []
    P_gradient = []
    update_gradient = []
    M_update_gradient = []

    Number_iterations = 0

    data1 = np.load('data/YaleB_train_3232.npy')
    data2 = torch.from_numpy(data1)
    data2 = data2.float()
    data3 = data2.to(torch.device("cuda:0"))

    LABELS = np.load('data/YaleB_train_gnd.npy')
    LABELS = torch.from_numpy(LABELS)
    LABELS = LABELS.long()
    LABELS = LABELS - 1
    # LABELS=LABELS.squeeze()
    LABELS = LABELS.to(torch.device("cuda:0"))
    LABELS = LABELS.squeeze()

    data_all = data3.view(batchsize_para, data3.shape[0] // batchsize_para, -1)
    LABELS = LABELS.view(batchsize_para, LABELS.shape[0] // batchsize_para)

    # adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(), lr=optimizer_lr)
    f_loss_optimizer = nn.MSELoss()

    RB = ReplayBuffer(500 * batchsize_para)

    Square = torch.eye(DIM)

    for i in range(Observe):
        RB.shuffle()
        if i == 0:
            M = torch.randn(batchsize_para, DIM, outputDIM).cuda()
            for k in range(batchsize_para):
                nn.init.orthogonal_(M[k])

            # M = (torch.randn(batchsize_para, DIM, outputDIM))
            state = (torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
                     torch.zeros(batchsize_para, outputDIM, outputDIM).cuda(),
                     torch.zeros(batchsize_para, DIM, DIM).cuda(),
                     )
            iteration = torch.zeros(batchsize_para)
            # M.retain_grad()
            M.requires_grad = True

            RB.push(state, M, iteration)
            count = 1
            print('observe finish', count)

        break_flag = False
        correct = 0
        total = 0
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            inputs = inputs.view(batchsize_para, inputs.shape[0] // batchsize_para, -1)
            labels = labels.view(batchsize_para, labels.shape[0] // batchsize_para)
            # inputs = inputs.permute(0, 2, 1)
            loss = f(inputs, labels, M.cuda())
            output = f_ac(inputs, M.cuda())
            output = output
            # output = torch.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))
            # loss = criterion(output, labels)
            # loss_total += loss.item()
            # loss_count += 1
            pred = output.argmax(dim=1).unsqueeze(0)
            total += labels.size(1)
            correct += torch.eq(pred, labels).sum().item()
            accu = round(100.0 * correct / total, 2)
            # loss_test = round(loss_total / loss_count, 4)
            print('~~~~~~~~~~~```````````````````````````````````````~accu', accu, 'epoch', i, 'correct', correct,
                  'total', total)
            loss.backward()
            # M_grad=M.grad
            M, state = hand_optimizee(M.grad, M, state)

            print('-------------------------')
            # print('MtM', torch.mm(M[k].t(),M[k]))

            iteration = iteration + 1
            for k in range(batchsize_para):
                if iteration[k] >= Optimizee_Train_Steps - opt.train_steps:
                    M[k] = Square[:, 0:outputDIM]
                    state[0][k] = torch.zeros(2, DIM, 20)
                    state[1][k] = torch.zeros(2, DIM, 20)
                    state[2][k] = torch.zeros(2, outputDIM, 20)
                    state[3][k] = torch.zeros(2, outputDIM, 20)
                    state[4][k] = torch.zeros(outputDIM, outputDIM)
                    state[5][k] = torch.zeros(DIM, DIM)
                    iteration[k] = 0

            state = (state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach(), state[4].detach(),
                     state[5].detach())
            M = M.detach()
            M.requires_grad = True
            M.retain_grad()

            RB.push(state, M, iteration)
            count = count + 1
            print('loss', loss.item() / opt.batchsize_data)
            print('observe finish', count)
            localtime = time.asctime(time.localtime(time.time()))

            if count >= Observe:
                break_flag = True
                break
        if break_flag == True:
            break

    RB.shuffle()

    check_point = optimizee.state_dict()
    check_point2 = optimizee.state_dict()
    check_point3 = optimizee.state_dict()
    Global_Epochs = 0
    train_svd = False
    total_loss = 0
    for i in range(Epochs):
        total_loss = 0
        print('\n=======> global training steps: {}'.format(i))
        if (i + 1) % Decay == 0 and (i + 1) != 0:
            count = count + 1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if opt.Imcrementflag == True:
            if (i + 1) % Imcrement == 0 and (i + 1) != 0:
                Optimizee_Train_Steps = Optimizee_Train_Steps + 50
        #
        if (i + 1) % opt.modelsave == 0 and (i + 1) != 0:
            print('-------------------------------SAVE----------------------------------------------')
            print(opt.modelsave)
            if opt.Pretrain == True:
                torch.save(optimizee.state_dict(),
                           'STATE' + str(i) + '_' + str(opt.optimizer_lr * 1000) + '_Decay' + str(
                               opt.Decay) + '_Observe' + str(opt.Observe) + '_Epochs' + str(
                               opt.Epochs) + '_Optimizee_Train_Steps' + str(
                               opt.Optimizee_Train_Steps) + '_train_steps' + str(
                               opt.train_steps) + '_hand_optimizer_lr' + str(opt.hand_optimizer_lr) + '.pth')
            else:
                torch.save(optimizee.state_dict(),
                           'STATE' + str(i) + '_' + str(opt.optimizer_lr * 1000) + '_Decay' + str(
                               opt.Decay) + '_Observe' + str(opt.Observe) + '_Epochs' + str(
                               opt.Epochs) + '_Optimizee_Train_Steps' + str(
                               opt.Optimizee_Train_Steps) + '_train_steps' + str(
                               opt.train_steps) + '_hand_optimizer_lr' + str(
                               opt.hand_optimizer_lr) + 'nopretrain_newlr_meanvar_devide2' + '.pth')

            # if opt.Pretrain==True:
            #     torch.save(optimizee.state_dict(), 'STATE/inner_epoch20_5/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')
            # else:
            #     torch.save(optimizee.state_dict(), 'epoch'+i+'.pth')

            # torch.save(optimizee.state_dict(), 'snapshot/'+str(i)+'_'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')

        # if i == 0:
        #     global_loss_graph = 0
        # else:
        #     if train_svd == False:
        #         global_loss_graph = global_loss_graph.detach()
        #         global_loss_graph = 0
        #     else:
        #         global_loss_graph = 0
        #         train_svd = False

        state_read, M_read, iteration_read = RB.sample(batchsize_para)
        state = (state_read[0].detach(), state_read[1].detach(), state_read[2].detach(), state_read[3].detach(),
                 state_read[4].detach(), state_read[5].detach())
        M = M_read.detach()
        iteration = iteration_read.detach()
        M.requires_grad = True
        M.retain_grad()

        flag = False
        break_flag = False
        count = 0
        new_count = 0
        begin = True
        adam_global_optimizer.zero_grad()
        if i == -1:
            with torch.no_grad():
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
                    # inputs = inputs.permute(0, 2, 1)
                    labels = labels - 1

                    output = f_ac(inputs, M.cuda())
                    output = output
                    output = torch.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))
                    # loss = criterion(output, labels)
                    # loss_total += loss.item()
                    loss_count += 1
                    pred = output.argmax(dim=1).unsqueeze(0)

                    correct += torch.eq(pred, labels).sum().item()
                    total += labels.size(1)
                accu = round(100.0 * correct / total, 2)
                loss_test = round(loss_total / loss_count, 4)
                print('~~~~~~~~~~~```````````````````````````````````````~accu', accu, 'epoch', i, 'correct', correct,
                      'total', total)
        start = time.time()
        while (1):
            correct = 0
            total = 0
            for j, data in enumerate(train_loader, 0):
                print('---------------------------------------------------------------------------')
                # print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).type(torch.LongTensor).cuda()
                inputs = inputs.view(batchsize_para, inputs.shape[0] // batchsize_para, -1)
                labels = labels.view(batchsize_para, labels.shape[0] // batchsize_para)

                # inputs = inputs.permute(0, 2, 1)
                with autograd.detect_anomaly():
                    loss = f(inputs, labels, M.cuda())
                    output = f_ac(inputs, M.cuda())
                    output = output
                    # output = torch.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))
                    # loss = criterion(output, labels)
                    # loss_total += loss.item()
                    # loss_count += 1
                    pred = output.argmax(dim=1).unsqueeze(0)
                    total += labels.size(1)

                    correct += torch.eq(pred, labels).sum().item()

                    loss.backward()
                total_loss = total_loss + loss
                accu = round(100.0 * correct / total, 2)
                # loss_test = round(loss_total / loss_count, 4)
                print('~~~~~~~~~~~```````````````````````````````````````~accu', accu, 'epoch', i, 'correct',
                      correct,
                      'total', total)
                print('loss', loss.item() / opt.batchsize_data)

                # #print('state',torch.sum(state[0]),torch.sum(state[1]),torch.sum(state[2]),torch.sum(state[3]))
                # M_grad=M.grad.data
                # P=M_grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),M_grad)
                # P=P*1e-4
                # print('EPOCHES:{},loss:{}'.format(i,loss.item()/640))
                # try:
                #     M_csgd=retraction(M,P,1)
                #     loss_csgd=f(data_all,M_csgd)
                #     print('EPOCHES:{},loss_csgd:{}'.format(i,loss_csgd.item()/60000))
                # except:
                #     print('svd')

                # print(inputs.shape)
                train_svd, M, next_state0, next_state1, next_state2, next_state3, next_state4, next_state5 = optimizee.step(
                    opt.s_lr, M, M.grad, state)
                state = (next_state0, next_state1, next_state2, next_state3, next_state4, next_state5)

                M = M.detach()
                M.requires_grad = True
                # pdb.set_trace()
                if count == opt.train_steps - 1:
                    M = M.detach()
                    M.requires_grad = True
                    validation_loss = compute_loss(M, data, batchsize_para)
                    M_validation_gradient = torch.autograd.grad(validation_loss, M, retain_graph=True)

                    M_validation_gradient_data = M_validation_gradient[0].data.detach()
                    vector = M_validation_gradient_data

                    loss_temp = f(inputs, labels,M)

                    # print('count',count,'loss2',loss2)
                    loss_temp.backward(retain_graph=True)
                    g_temp = M.grad
                    g_temp = Variable(g_temp)
                    g_temp.requires_grad = True

                    # loss_temp = f(inputs, labels, M.cuda())
                    s_i = M
                    # g_temp = torch.autograd.grad(loss_temp, M)[0]
                    M_grad = g_temp

                    s_i_shape = M.shape
                    # M_grad = s_i.grad
                    M_grad = M_grad.squeeze().cuda()

                    M = s_i.squeeze().cuda()
                    Mshape = M.shape
                    G = othogonal_projection(M, M_grad)
                    G = G * s_lr
                    GGt = G.mm(G.T)
                    L_input = optimizee.LogAndSign_Preprocess_Gradient(GGt)
                    GtG = G.T.mm(G)
                    R_input = optimizee.LogAndSign_Preprocess_Gradient(GtG)

                    L_h0_i = state[0].squeeze().cuda()  # cuda()
                    L_c0_i = state[1].squeeze().cuda()  # cuda()
                    R_h0_i = state[2].squeeze().cuda()  # cuda()
                    R_c0_i = state[3].squeeze().cuda()
                    L_before = state[5].squeeze().cuda()  # cuda()
                    R_before = state[4].squeeze().cuda()  # cuda()

                    l_output, (hn_l, cn_l) = optimizee.lstm(L_input, (L_h0_i, L_c0_i))
                    r_output, (hn_r, cn_r) = optimizee.lstm(R_input, (R_h0_i, R_c0_i))
                    l_ = optimizee.L_linear(
                        l_output.squeeze()) * optimizee.output_scale  # 因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上
                    l_ = l_ - l_.mean() + 1.0
                    L = l_.squeeze().diag()
                    # L = torch.diag_embed(L)
                    r_ = optimizee.R_linear(r_output.squeeze()) * optimizee.output_scale
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
                    update_R = P.unsqueeze(0)
                    try:
                        M_end = retraction(M, -1e-7*update_R)

                    except:

                       print('svd')
                       break_flag = True
                       break
                    # vector = g_temp.detach().clone()
                    Jacobi_vector = vector
                    vector_temp_list = []

                    for i in range(opt.Neumann_series):
                        # pdb.set_trace()
                        vector_temp = torch.autograd.grad(M_end, update_R, grad_outputs=vector, retain_graph=True)[0]

                        # vector_temp_list.append(vector_temp.detach().cpu().numpy())

                        vector_temp = torch.autograd.grad(update_R, M, grad_outputs=vector_temp, retain_graph=True)[
                            0].unsqueeze(0)
                        vector_temp2 = torch.autograd.grad(M_end, M, grad_outputs=vector, retain_graph=True)[0]

                        vector = vector -opt.Neumann_alpha * (vector_temp + vector_temp2)

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

                    update_R_optimizee = torch.autograd.grad(update_R, optimizee.parameters(),
                                                             grad_outputs=vector_M_update, retain_graph=True)

                    check = 0
                    for p in optimizee.parameters():
                        check = check + 1 if type(p.grad) == type(None) else check
                    if check > 0:
                        back_loss = optimizee_loss(optimizee)

                        back_loss.backward()
                    for i, p in enumerate(optimizee.parameters()):
                        p.grad = update_R_optimizee[i]


                    adam_global_optimizer.step()
                    end = time.time()
                    print('time:', end - start)
                    pdb.set_trace()

                    # optimizer_iteration = optimizer_iteration + 1
                    break_flag = True

                    iteration = iteration + 1
                    # pdb.set_trace()

                    break

                count = count + 1


            if break_flag == True:
                break

                # P=M_grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),M_grad)

        iteration = iteration + 1
        end = time.time()
        print('time:', end - start)
        pdb.set_trace()

        # M.retain_grad()
        # print(M.requires_grad)
        # M.requires_grad=True

        # global_loss_graph = f(data_all, LABELS, M.cuda())

        # loss.backward(retain_graph=True)

        # global_loss_graph.backward()
        print('total_loss', (total_loss / count) / opt.batchsize_data)
        # pdb.set_trace()
        for k in range(batchsize_para):
            if iteration[k] >= Optimizee_Train_Steps - opt.train_steps:
                M[k] = Square[:, 0:outputDIM]
                state[0][k] = torch.zeros(2, DIM, 20).cuda()
                state[1][k] = torch.zeros(2, DIM, 20).cuda()
                state[2][k] = torch.zeros(2, outputDIM, 20).cuda()
                state[3][k] = torch.zeros(2, outputDIM, 20).cuda()
                state[4][k] = torch.zeros(outputDIM, outputDIM).cuda()
                state[5][k] = torch.zeros(DIM, DIM).cuda()

        RB.push((state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach(), state[4].detach(),
                 state[5].detach()), M.detach().unsqueeze(0),
                iteration.detach())

        print('==========>EPOCHES<-=========', i)
        # print('=======>global_loss_graph', global_loss_graph.item() / 60000)
        # writer.add_scalars('global_loss', {'global_loss': global_loss_graph.item() / 60000}, i)
        # writer.add_scalars('global_loss_log', {'global_loss': math.log(global_loss_graph.item() / 60000)}, i)

    # writer.close()
