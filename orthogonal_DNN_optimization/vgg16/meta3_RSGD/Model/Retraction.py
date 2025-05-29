import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function

from Model.Eiglayer import EigLayer
from Model.mysqrt import M_Sqrt


class Retraction(nn.Module):
    def __init__(self, lr):
        super(Retraction, self).__init__()

        # self.beta=lr
        self.msqrt2=M_Sqrt(-1)
        self.eiglayer1=EigLayer()


    def forward(self, inputs, grad,lr):


        new_point=torch.zeros(inputs.shape)#.cuda()
        n=inputs.shape[0]

        P=-lr*grad
        A=inputs+P #A(t)

        A_p = A.T.mm(A) #A'(t)

        S, U = self.eiglayer1(A_p)
        S_2 = self.msqrt2(S)
        A_uf = torch.matmul(torch.matmul(U, S_2), U.T)

        new_point = torch.matmul(A, A_uf)


        return new_point