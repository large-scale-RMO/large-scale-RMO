import torch
import torch.nn as nn
import torch.nn.functional as F


class vgg16_BN(nn.Module):
    def __init__(self):
        super(vgg16_BN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, param, bn_training=True):
        e_params=param['e_params']
        s_params=param['s_params']
        bn_param = param['bn_params']
        # conv1
        x = F.conv2d(x, e_params[0], e_params[1], padding=1)
        x = F.batch_norm(x, bn_param[0], bn_param[1], e_params[2], e_params[3], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[0], e_params[4], padding=1)
        x = F.batch_norm(x, bn_param[2], bn_param[3], e_params[5], e_params[6], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv2
        x = F.conv2d(x, s_params[1], e_params[7], padding=1)
        x = F.batch_norm(x, bn_param[4], bn_param[5], e_params[8], e_params[9], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x,s_params[2], e_params[10], padding=1)
        x = F.batch_norm(x, bn_param[6], bn_param[7], e_params[11], e_params[12], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv3
        x = F.conv2d(x, s_params[3], e_params[13], padding=1)
        x = F.batch_norm(x, bn_param[8], bn_param[9], e_params[14], e_params[15], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[4], e_params[16], padding=1)
        x = F.batch_norm(x, bn_param[10], bn_param[11], e_params[17], e_params[18], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[5], e_params[19], padding=1)
        x = F.batch_norm(x, bn_param[12], bn_param[13], e_params[20], e_params[21], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv4
        x = F.conv2d(x, s_params[6], e_params[22], padding=1)
        x = F.batch_norm(x, bn_param[14], bn_param[15], e_params[23], e_params[24], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[7], e_params[25], padding=1)
        x = F.batch_norm(x, bn_param[16], bn_param[17], e_params[26], e_params[27], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[8], e_params[28], padding=1)
        x = F.batch_norm(x, bn_param[18], bn_param[19], e_params[29], e_params[30], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # conv5
        x = F.conv2d(x, s_params[9], e_params[31], padding=1)
        x = F.batch_norm(x, bn_param[20], bn_param[21], e_params[32], e_params[33], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[10], e_params[34], padding=1)
        x = F.batch_norm(x, bn_param[22], bn_param[23], e_params[35], e_params[36], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_params[11], e_params[37], padding=1)
        x = F.batch_norm(x, bn_param[24], bn_param[25], e_params[38], e_params[39], training=bn_training)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2, 2)

        # classifier
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # print(e_params[14].shape)
        x = F.linear(x, e_params[40], e_params[41])
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = F.linear(x, e_params[42], e_params[43])
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = F.linear(x, e_params[44], e_params[45])
        return x


