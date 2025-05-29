import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function
from Model.Eiglayer import EigLayer
from Model.mysqrt import M_Sqrt


class Retraction(nn.Module):
    def __init__(self,lr):
        super(Retraction, self).__init__()

        self.beta=lr
        self.msqrt2=M_Sqrt(-1)
        self.eiglayer1=EigLayer()


    def forward(self, inputs, P):


        new_point=torch.zeros(inputs.shape).cuda()
        n=inputs.shape[0]

        # P=-lr*grad
        PV=inputs+P #A(t)

        PV_p=torch.matmul(PV.T,PV) #A'(t)

        # PV_S,PV_U=self.eiglayer1(PV_p)
        # PV_S2=self.msqrt2(PV_S)
        # PV_p=torch.matmul( torch.matmul( PV_U, PV_S2 ), PV_U.T )
        PV_m = self.msqrt2(PV_p)

        new_point=torch.matmul(PV,PV_m)

        return new_point