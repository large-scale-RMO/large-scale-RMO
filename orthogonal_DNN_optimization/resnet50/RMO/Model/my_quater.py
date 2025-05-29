import torch
import torch.nn as nn

class My_Quater(nn.Module):
    def __init__(self, sign):
        super(My_Quater, self).__init__()
        self.sign = sign
        self.epsilon = 0.0000001

    def forward(self, input):
        one = torch.ones(input.shape).cuda()

        output = input
        miner = -1 * input
        # output = torch.where(output > 0, output, self.epsilon * one)
        output = torch.where(output > 0, output, miner)
        output = torch.where(output > 0, output, self.epsilon * one)
        output = torch.pow(output, -0.25)
        miner = -1 * output
        output = torch.where(input >= 0, output, miner)


        return output
