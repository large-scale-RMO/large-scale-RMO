import torch
import torch.nn as nn

from learning_to_learn_ import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from hand_optimizer.handcraft_optimizer import Hand_Optimizee_Model
from DataSet.YaleB import YaleB
from rsamo import meta_optimizer
# from utils import FastRandomIdentitySampler

import config

opt = config.parse_opt()
print(opt)
print('-----------------------------------')
print(opt.train_steps)
LSTM_Optimizee = meta_optimizer(opt).cuda()

if opt.Pretrain==True:
    '''
    checkpoint2 = torch.load(opt.prepath2)
    LSTM_Optimizee.load_state_dict(checkpoint2,strict=False)
    checkpoint = torch.load(opt.prepath)
    LSTM_Optimizee.load_state_dict(checkpoint,strict=False)
    '''
    checkpoint=torch.load(opt.prepath)
    LSTM_Optimizee.load_state_dict(checkpoint,strict=False)


#checkpoint2 = torch.load(opt.prepath)
#LSTM_Optimizee.load_state_dict(checkpoint2)

Hand_Optimizee = Hand_Optimizee_Model(opt.hand_optimizer_lr)

print('pretrain finsist')

train_mnist = YaleB(opt.datapath, train=False)
test_mnist = YaleB(opt.datapath, train=False)
'''
train_loader = torch.utils.data.DataLoader(
    train_mnist, batch_size=opt.batchsize_data,
    sampler=FastRandomIdentitySampler(train_mnist, num_instances=opt.num_instances),
    drop_last=True, pin_memory=True, num_workers=opt.nThreads)
'''

train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=True, num_workers=0)

train_test_loader = torch.utils.data.DataLoader(
        test_mnist, batch_size=8,shuffle=True, drop_last=True, num_workers=0)


global_loss_list ,flag = Learning_to_learn_global_training(opt,Hand_Optimizee,LSTM_Optimizee,train_loader,train_test_loader)
