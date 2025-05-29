import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function


class EigLayerF(Function):
    @staticmethod
    def forward(self, input):
        n = input.shape[0]  # input的高度
        S = torch.zeros(input.shape).cuda()
        U = torch.zeros(input.shape).cuda()

        for i in range(n):
            value, vector = torch.linalg.eig(input[i])
            S[i] = torch.diag(value[:])  # input[i]的特征值组成的对角矩阵
            U[i] = vector  # input[i]的特征向量矩阵

        self.save_for_backward(input, S, U)
        # print('EigLayer finish')
        return S, U

    @staticmethod
    def backward(self, grad_S, grad_U):
        # print('grad_S',torch.sum(grad_S))
        # print('grad_U',torch.sum(grad_U))

        input, S, U = self.saved_tensors
        n = input.shape[0]
        dim = input.shape[1]

        grad_input = V(torch.zeros(input.shape)).cuda()  # grad_input初始化为0

        e = torch.eye(dim).cuda()  # e=Idim

        P_i = torch.matmul(S, torch.ones(dim, dim).cuda())
        # print('P_i',P_i)
        # print('P_i sum',torch.sum(P_i))

        # P=(P_i-P_i.permute(0,2,1))+e+0.00001-0.00001*e

        P = (P_i - P_i.permute(0, 2,
                               1)) + e  ###########################################################################?????
        epo = (torch.ones(P.shape)).cuda() * 0.000001#

        P = torch.where(P != 0, P, epo)  # 对于每个元素P!=0,则取P的对应元素，否则取epo的对应元素
        # print('P',P)

        P = (1 / P) - e
        # print('P',P)
        # print('1/P sum',torch.sum(P))

        g1 = torch.matmul(U.permute(0, 2, 1), grad_U)
        g1 = (g1 + g1.permute(0, 2, 1)) / 2
        g1 = torch.mul(P.permute(0, 2, 1), g1)
        g1 = 2 * torch.matmul(torch.matmul(U, g1), U.permute(0, 2, 1))

        g2 = torch.matmul(torch.matmul(U, torch.mul(grad_S, e)), U.permute(0, 2, 1))

        grad_input = g1 + g2

        # print('grad_input',torch.sum(grad_input))

        return grad_input


class EigLayer(nn.Module):
    def __init__(self):
        super(EigLayer, self).__init__()

    def forward(self, input1):
        return EigLayerF().apply(input1)

