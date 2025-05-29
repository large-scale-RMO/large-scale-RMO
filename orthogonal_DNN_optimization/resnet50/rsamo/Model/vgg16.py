import torch
import torch.nn as nn
import torch.nn.functional as F


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, param):
        e_params=param['e_params']
        s_params=param['s_params']
        # conv1
        x = F.conv2d(x, e_params[0], e_params[1], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[0], e_params[2], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv2
        x = F.conv2d(x, s_params[1], e_params[3], padding=1)
        x = F.relu(x)
        x = F.conv2d(x,s_params[2], e_params[4], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv3
        x = F.conv2d(x, s_params[3], e_params[5], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[4], e_params[6], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[5], e_params[7], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv4
        x = F.conv2d(x, s_params[6], e_params[8], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[7], e_params[9], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[8], e_params[10], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv5
        x = F.conv2d(x, s_params[9], e_params[11], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[10], e_params[12], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, s_params[11], e_params[13], padding=1)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2, 2)

        # classifier
        x = x.view(x.size(0), -1)
        x = F.linear(x, e_params[14], e_params[15])
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = F.linear(x, e_params[16], e_params[17])
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = F.linear(x, e_params[18], e_params[19])
        return x


