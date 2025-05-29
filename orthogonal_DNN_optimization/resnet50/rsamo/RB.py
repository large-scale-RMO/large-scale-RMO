import random

import torch
from collections import deque

class RB(object):
    def __init__(self,Replay_size):
        self.maxlen = Replay_size
        self.buffer = deque(maxlen=Replay_size)

    def sample(self, batchsize):
        params = random.sample(self.buffer, batchsize)
        new_params = []
        for param in params:
            new_e_params = []
            new_s_params = []
            new_bn_params = []
            new_L_h0 = []
            new_R_h0 = []
            new_L_c0 = []
            new_R_c0 = []
            new_L_be = []
            new_R_be = []
            for e_i in param['e_params']:
                new_e_params.append(e_i.detach())
            for s_i in param['s_params']:
                new_s_params.append(s_i.detach())
            for b_i in param['bn_params']:
                new_bn_params.append(b_i.detach())
            for L_h0_i in param['L_h0']:
                new_L_h0.append(L_h0_i.detach())
            for L_c0_i in param['L_c0']:
                new_L_c0.append(L_c0_i.detach())
            for R_h0_i in param['R_h0']:
                new_R_h0.append(R_h0_i.detach())
            for R_c0_i in param['R_c0']:
                new_R_c0.append(R_c0_i.detach())
            for L_be_i in param['L_before']:
                new_L_be.append(L_be_i.detach())
            for R_be_i in param['R_before']:
                new_R_be.append(R_be_i.detach())


            new_param = {
                'e_params': new_e_params,
                's_params': new_s_params,
                'bn_params': new_bn_params,
                'L_h0': new_L_h0,
                'L_c0': new_L_c0,
                'R_h0': new_R_h0,
                'R_c0': new_R_c0,
                'L_before': new_L_be,
                'R_before': new_R_be
            }
            new_params.append(new_param)
        return new_params


    def shuffle(self):
        random.shuffle(self.buffer)

    def push(self, param):
        self.buffer.append(param)

    def get_length(self):
        return len(self.buffer)

    def is_full(self):
        if len(self.buffer) < self.maxlen:
            return 0
        else:
            return 1
