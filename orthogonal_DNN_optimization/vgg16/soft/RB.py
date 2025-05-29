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
            new_bn_params = []

            new_R_c0 = []
            for e_i in param['e_params']:
                new_e_params.append(e_i.detach())
            for b_i in param['bn_params']:
                new_bn_params.append(b_i.detach())


            new_param = {
                'e_params': new_e_params,
                'bn_params': new_bn_params,
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
