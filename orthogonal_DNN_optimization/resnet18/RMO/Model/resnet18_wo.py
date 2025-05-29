import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, param, bn_training=True):
        e_param = param['e_params']
        s_param = param['s_params']
        bn_param = param['bn_params']

        # conv1
        x = F.conv2d(x, s_param[0], stride=2, padding=3)
        x = F.batch_norm(x, bn_param[0], bn_param[1], e_param[0], e_param[1], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        #block1
        res = x
        x = F.conv2d(x, s_param[1], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[2], bn_param[3], e_param[2], e_param[3], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[2], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[4], bn_param[5], e_param[4], e_param[5], training=bn_training)
        x += res
        # x = F.relu(x)

        #block2
        res = x
        x = F.conv2d(x, s_param[3], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[6], bn_param[7], e_param[6], e_param[7], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[4], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[8], bn_param[9], e_param[8], e_param[9], training=bn_training)
        x += res
        # x = F.relu(x)

        #block3
        res = x
        x = F.conv2d(x, s_param[5], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[10], bn_param[11], e_param[10], e_param[11], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[6], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[12], bn_param[13], e_param[12], e_param[13], training=bn_training)
        res = F.conv2d(res, e_param[14], stride=2, padding=0)
        res = F.batch_norm(res, bn_param[14], bn_param[15], e_param[15], e_param[16], training=bn_training)
        x += res
        # x = F.relu(x)

        #block4
        res = x
        x = F.conv2d(x, s_param[7], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[16], bn_param[17], e_param[17], e_param[18], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[8], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[18], bn_param[19], e_param[19], e_param[20], training=bn_training)
        x += res
        # x = F.relu(x)

        #block5
        res = x
        x = F.conv2d(x, s_param[9], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[20], bn_param[21], e_param[21], e_param[22], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[10], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[22], bn_param[23], e_param[23], e_param[24], training=bn_training)

        res = F.conv2d(res, e_param[25], stride=2, padding=0)
        res = F.batch_norm(res, bn_param[24], bn_param[25], e_param[26], e_param[27], training=bn_training)
        x += res
        # x = F.relu(x)

        #block6
        res = x
        x = F.conv2d(x, s_param[11], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[26], bn_param[27], e_param[28], e_param[29], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[12], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[28], bn_param[29], e_param[30], e_param[31], training=bn_training)
        x += res
        # x = F.relu(x)

        #block7
        res = x
        x = F.conv2d(x, s_param[13], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[30], bn_param[31], e_param[32], e_param[33], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[14], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[32], bn_param[33], e_param[34], e_param[35], training=bn_training)
        res = F.conv2d(res, e_param[36], stride=2, padding=0)
        res = F.batch_norm(res, bn_param[34], bn_param[35], e_param[37], e_param[38], training=bn_training)
        x += res
        # x = F.relu(x)

        #block8
        res = x
        x = F.conv2d(x, s_param[15], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[36], bn_param[37], e_param[39], e_param[40], training=bn_training)
        x = F.relu(x)
        x = F.conv2d(x, s_param[16], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[38], bn_param[39], e_param[41], e_param[42], training=bn_training)
        x += res
        # x = F.relu(x)

        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = F.linear(x, e_param[43], e_param[44])

        return x


