import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function


class EigLayerF(Function):
    @staticmethod
    def forward(self, input):
        S = torch.zeros(input.shape).cuda()
        U = torch.zeros(input.shape).cuda()

        value, vector = torch.eig(input, eigenvectors=True)
        S = torch.diag(value[:, 0])
        U = vector

        self.save_for_backward(input, S, U)
        # print('EigLayer finish')
        return S, U

    @staticmethod
    def backward(self, grad_S, grad_U):


        input, S, U = self.saved_tensors
        # n = input.shape[0]
        dim = input.shape[1]

        grad_input = V(torch.zeros(input.shape)).cuda()  # grad_input初始化为0

        e = torch.eye(dim).cuda()  # e=Idim

        P_i = torch.matmul(S, torch.ones(dim, dim).cuda())


        P = (P_i - P_i.T) + e
        epo = (torch.ones(P.shape).cuda()) * 0.000001

        P = torch.where(P != 0, P, epo)  # 对于每个元素P!=0,则取P的对应元素，否则取epo的对应元素
        P = (1 / P) - e


        g1 = torch.matmul(U.T, grad_U)
        g1 = (g1 + g1.T) / 2
        g1 = torch.mul(P.T, g1)
        g1 = 2 * torch.matmul(torch.matmul(U, g1), U.T)

        g2 = torch.matmul(torch.matmul(U, torch.mul(grad_S, e)), U.T)

        grad_input = g1 + g2


        return grad_input


class EigLayer(nn.Module):
    def __init__(self):
        super(EigLayer, self).__init__()

    def forward(self, input1):
        return EigLayerF().apply(input1)

    # def backward(self):
    #     return EigLayerF.apply()

