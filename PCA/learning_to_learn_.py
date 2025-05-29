import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import time
import math, random

import numpy as np 

# from losses.LOSS import ContrastiveLoss
from ReplyBuffer import ReplayBuffer
from retraction import Retraction

retraction=Retraction(1)





def f(inputs,M):
    
    X=torch.matmul(M,M.permute(0,2,1)).cuda()
    X2=torch.matmul(X,inputs)
    L=torch.norm(inputs-X2,dim=1).pow(2)
    L=torch.sum(L)

    return L



def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,hand_optimizee,optimizee,train_loader):
    writer = SummaryWriter('./log')
    DIM=opt.DIM
    outputDIM=opt.outputDIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    Epochs=opt.Epochs
    s_lr = opt.s_lr
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate
    Imcrement=opt.Imcrement
    Sample_number=opt.Sample_number
    X=[]
    Y=[]
    Number_iterations=0


    data1=np.load('data/dim784_training_images_bool.npy')
    data2=torch.from_numpy(data1)
    data2=data2.float()
    data3=data2.to(torch.device("cuda:0"))

    data_all=data3.view(batchsize_para,data3.shape[0]//batchsize_para,-1)
    data_all=data_all.permute(0,2,1)

    #adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)

    RB=ReplayBuffer(500*batchsize_para)

    Square=torch.eye(DIM)

    for i in range(Observe):
        RB.shuffle()
        if i ==0:
            M=torch.randn(batchsize_para,DIM, outputDIM).cuda()
            for k in range(batchsize_para):
                nn.init.orthogonal_(M[k])

            # M = (torch.randn(batchsize_para, DIM, outputDIM))
            state = (torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, DIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
                     torch.zeros(batchsize_para, 2, outputDIM, 20).cuda(),
                     torch.zeros(batchsize_para, outputDIM, outputDIM).cuda(),
                     torch.zeros(batchsize_para,  DIM, DIM).cuda(),
                     )
            iteration=torch.zeros(batchsize_para)
            #M.retain_grad()
            M.requires_grad=True

            RB.push(state,M,iteration)  
            count=1
            print ('observe finish',count)

        break_flag=False
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
            labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)
            inputs=inputs.permute(0,2,1)
            loss = f(inputs,M)
            
            loss.backward()
            #M_grad=M.grad
            M, state = hand_optimizee(M.grad, M, state)

            print('-------------------------')
            #print('MtM', torch.mm(M[k].t(),M[k]))

            iteration=iteration+1
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:
                    M[k]=Square[:,0:outputDIM].cuda()
                    state[0][k]=torch.zeros(2, DIM, 20).cuda()
                    state[1][k]=torch.zeros(2, DIM, 20).cuda()
                    state[2][k]=torch.zeros(2, outputDIM, 20).cuda()
                    state[3][k]=torch.zeros(2, outputDIM, 20).cuda()
                    state[4][k] = torch.zeros(outputDIM, outputDIM).cuda()
                    state[5][k] = torch.zeros(DIM, DIM).cuda()  
                    iteration[k]=0


            state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach(),state[4].detach(),state[5].detach())
            M=M.detach()
            M.requires_grad=True
            M.retain_grad()
            

            RB.push(state,M,iteration)
            count=count+1
            print ('loss',loss.item()/opt.batchsize_data)
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )

            if count>=Observe:
                break_flag=True
                break
        if break_flag==True:
            break                         
    
    RB.shuffle()

    train_svd=False
    total_loss  = 0
    for i in range(Epochs):
        total_loss=0
        print('\n=======> global training steps: {}'.format(i))
        if (i+1) % Decay==0 and (i+1) != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)
            s_lr = Decay_rate* s_lr

        if opt.Imcrementflag==True:
            if (i+1) % Imcrement==0 and (i+1) != 0:
                Optimizee_Train_Steps=Optimizee_Train_Steps+50
        #
        if (i+1) % opt.modelsave==0 and (i+1) != 0:
            print('-------------------------------SAVE----------------------------------------------')
            print(opt.modelsave)
            torch.save(optimizee.state_dict(), 'new_new'+str(9998+i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'nopretrain_newlr_meanvar_devide2'+'.pth')
            # if opt.Pretrain==True:
            #     torch.save(optimizee.state_dict(), 'STATE/inner_epoch20_5/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')
            # else:
            #     torch.save(optimizee.state_dict(), 'epoch'+i+'.pth')

            # torch.save(optimizee.state_dict(), 'snapshot/'+str(i)+'_'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')


        if i==0:
            global_loss_graph=0
        else:
            if train_svd==False:
                global_loss_graph=global_loss_graph.detach()
                global_loss_graph=0
            else:
                global_loss_graph=0
                train_svd=False


        state_read, M_read, iteration_read= RB.sample(batchsize_para)
        state=(state_read[0].detach(),state_read[1].detach(),state_read[2].detach(),state_read[3].detach(),state_read[4].detach(),state_read[5].detach())
        M=M_read.detach()        
        iteration=iteration_read.detach()
        M.requires_grad=True
        M.retain_grad()


        flag=False
        break_flag=False
        count=0
        new_count=0
        begin=True
        adam_global_optimizer.zero_grad()
        while(1):
            for j, data in enumerate(train_loader, 0):
                print('---------------------------------------------------------------------------')
                #print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
                labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)

                inputs=inputs.permute(0,2,1)


                loss = f(inputs,M)
                total_loss+=loss

                loss.backward(retain_graph=True)
                print('loss',loss.item()/opt.batchsize_data)

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




                #print(inputs.shape)
                train_svd, M, next_state0, next_state1, next_state2, next_state3, next_state4, next_state5 = optimizee.step(s_lr,M,M.grad,state)
                state = (next_state0, next_state1, next_state2, next_state3, next_state4, next_state5)
 

                count=count+1


                if count==opt.train_steps:
                    break_flag=True
                    break
            if break_flag==True:
                break

                    #P=M_grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),M_grad)

        iteration=iteration+1

        # M.retain_grad()
        #print(M.requires_grad)
        #M.requires_grad=True

        global_loss_graph= f(data_all,M)



        # loss.backward(retain_graph=True)


        # global_loss_graph.backward()
        total_loss.backward()
        adam_global_optimizer.step()
        print('total_loss',(total_loss/count)/opt.batchsize_data)


        for k in range(batchsize_para):
            if iteration[k] >= Optimizee_Train_Steps - opt.train_steps:
                M[k] = Square[:, 0:outputDIM]
                state[0][k] = torch.zeros(2, DIM, 20).cuda()
                state[1][k] = torch.zeros(2, DIM, 20).cuda()
                state[2][k] = torch.zeros(2, outputDIM, 20).cuda()
                state[3][k] = torch.zeros(2, outputDIM, 20).cuda()
                state[4][k] = torch.zeros(outputDIM, outputDIM).cuda()
                state[5][k] = torch.zeros(DIM, DIM).cuda()

        RB.push((state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach(),state[4].detach(),state[5].detach()), M.detach(),
                iteration.detach())


        print('==========>EPOCHES<-=========',i)
        print('=======>global_loss_graph',global_loss_graph.item()/60000)
        writer.add_scalars('global_loss', {'global_loss': global_loss_graph.item()/60000}, i)
        writer.add_scalars('global_loss_log', {'global_loss': math.log(global_loss_graph.item() / 60000)}, i)


    writer.close()