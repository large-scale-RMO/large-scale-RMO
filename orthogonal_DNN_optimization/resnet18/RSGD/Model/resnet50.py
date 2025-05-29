import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
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

        # layer1
        # conv_block1
        res = x
        x = F.conv2d(x, s_param[1])
        x = F.batch_norm(x, bn_param[2], bn_param[3], e_param[2], e_param[3], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[2], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[4], bn_param[5], e_param[4], e_param[5], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[6])
        x = F.batch_norm(x, bn_param[6], bn_param[7], e_param[7], e_param[8], training=bn_training)

        res = F.conv2d(res, e_param[9], stride=1)
        res = F.batch_norm(res, bn_param[8], bn_param[9], e_param[10], e_param[11], training=bn_training)

        x += res
        x = F.relu(x)

        # identity_block1_1
        res = x
        x = F.conv2d(x, s_param[3])
        x = F.batch_norm(x, bn_param[10], bn_param[11], e_param[12], e_param[13], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[4], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[12], bn_param[13], e_param[14], e_param[15], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[16])
        x = F.batch_norm(x, bn_param[14], bn_param[15], e_param[17], e_param[18], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block1_2
        res = x
        x = F.conv2d(x, s_param[5])
        x = F.batch_norm(x, bn_param[16], bn_param[17], e_param[19], e_param[20], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[6], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[18], bn_param[19], e_param[21], e_param[22], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[23])
        x = F.batch_norm(x, bn_param[20], bn_param[21], e_param[24], e_param[25], training=bn_training)
        x += res
        x = F.relu(x)

        # layer2
        # conv_block2
        res = x
        x = F.conv2d(x, s_param[7])
        x = F.batch_norm(x, bn_param[22], bn_param[23], e_param[26], e_param[27], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[8], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[24], bn_param[25], e_param[28], e_param[29], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[30])
        x = F.batch_norm(x, bn_param[26], bn_param[27], e_param[31], e_param[32], training=bn_training)

        res = F.conv2d(res, e_param[33], stride=2)
        res = F.batch_norm(res, bn_param[28], bn_param[29], e_param[34], e_param[35], training=bn_training)

        x += res
        x = F.relu(x)

        # identity_block2_1
        res = x
        x = F.conv2d(x, s_param[9])
        x = F.batch_norm(x, bn_param[30], bn_param[31], e_param[36], e_param[37], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[10], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[32], bn_param[33], e_param[38], e_param[39], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[40])
        x = F.batch_norm(x, bn_param[34], bn_param[35], e_param[41], e_param[42], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block2_2
        res = x
        x = F.conv2d(x, s_param[11])
        x = F.batch_norm(x, bn_param[36], bn_param[37], e_param[43], e_param[44], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[12], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[38], bn_param[39], e_param[45], e_param[46], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[47])
        x = F.batch_norm(x, bn_param[40], bn_param[41], e_param[48], e_param[49], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block2_3
        res = x
        x = F.conv2d(x, s_param[13])
        x = F.batch_norm(x, bn_param[42], bn_param[43], e_param[50], e_param[51], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[14], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[44], bn_param[45], e_param[52], e_param[53], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[54])
        x = F.batch_norm(x, bn_param[46], bn_param[47], e_param[55], e_param[56], training=bn_training)
        x += res
        x = F.relu(x)

        # layer3
        # conv_block3
        res = x
        x = F.conv2d(x, s_param[15])
        x = F.batch_norm(x, bn_param[48], bn_param[49], e_param[57], e_param[58], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[16], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[50], bn_param[51], e_param[59], e_param[60], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[61])
        x = F.batch_norm(x, bn_param[52], bn_param[53], e_param[62], e_param[63], training=bn_training)

        res = F.conv2d(res, e_param[64], stride=2)
        res = F.batch_norm(res, bn_param[54], bn_param[55], e_param[65], e_param[66], training=bn_training)

        x += res
        x = F.relu(x)

        # identity_block3_1
        res = x
        x = F.conv2d(x, s_param[17])
        x = F.batch_norm(x, bn_param[56], bn_param[57], e_param[67], e_param[68], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[18], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[58], bn_param[59], e_param[69], e_param[70], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[71])
        x = F.batch_norm(x, bn_param[60], bn_param[61], e_param[72], e_param[73], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block3_2
        res = x
        x = F.conv2d(x, s_param[19] )
        x = F.batch_norm(x, bn_param[62], bn_param[63], e_param[74], e_param[75], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[20], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[64], bn_param[65], e_param[76], e_param[77], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[78] )
        x = F.batch_norm(x, bn_param[66], bn_param[67], e_param[79], e_param[80], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block3_3
        res = x
        x = F.conv2d(x, s_param[21] )
        x = F.batch_norm(x, bn_param[68], bn_param[69], e_param[81], e_param[82], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[22], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[70], bn_param[71], e_param[83], e_param[84], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[85] )
        x = F.batch_norm(x, bn_param[72], bn_param[73], e_param[86], e_param[87], training=bn_training)
        x += res
        x = F.relu(x)
        # identity_block3_4
        res = x
        x = F.conv2d(x, s_param[23] )
        x = F.batch_norm(x, bn_param[74], bn_param[75], e_param[88], e_param[89], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[24], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[76], bn_param[77], e_param[90], e_param[91], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[92] )
        x = F.batch_norm(x, bn_param[78], bn_param[79], e_param[93], e_param[94], training=bn_training)
        x += res
        x = F.relu(x)
        # identity_block3_5
        res = x
        x = F.conv2d(x, s_param[25] )
        x = F.batch_norm(x, bn_param[80], bn_param[81], e_param[95], e_param[96], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[26], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[82], bn_param[83], e_param[97], e_param[98], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[99] )
        x = F.batch_norm(x, bn_param[84], bn_param[85], e_param[100], e_param[101], training=bn_training)
        x += res
        x = F.relu(x)

        # layer4
        # conv_block4
        res = x
        x = F.conv2d(x, s_param[27] )
        x = F.batch_norm(x, bn_param[86], bn_param[87], e_param[102], e_param[103], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[28], stride=2, padding=1)
        x = F.batch_norm(x, bn_param[88], bn_param[89], e_param[104], e_param[105], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[106] )
        x = F.batch_norm(x, bn_param[90], bn_param[91], e_param[107], e_param[108], training=bn_training)

        res = F.conv2d(res, e_param[109], stride=2)
        res = F.batch_norm(res, bn_param[92], bn_param[93], e_param[110], e_param[111], training=bn_training)

        x += res
        x = F.relu(x)

        # identity_block4_1
        res = x
        x = F.conv2d(x, s_param[29] )
        x = F.batch_norm(x, bn_param[94], bn_param[95], e_param[112], e_param[113], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[30], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[96], bn_param[97], e_param[114], e_param[115], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[116] )
        x = F.batch_norm(x, bn_param[98], bn_param[99], e_param[117], e_param[118], training=bn_training)
        x += res
        x = F.relu(x)

        # identity_block4_2
        res = x
        x = F.conv2d(x, s_param[31] )
        x = F.batch_norm(x, bn_param[100], bn_param[101], e_param[119], e_param[120], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, s_param[32], stride=1, padding=1)
        x = F.batch_norm(x, bn_param[102], bn_param[103], e_param[121], e_param[122], training=bn_training)
        x = F.relu(x)

        x = F.conv2d(x, e_param[123])
        x = F.batch_norm(x, bn_param[104], bn_param[105], e_param[124], e_param[125], training=bn_training)
        x += res
        x = F.relu(x)

        # fc
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = F.linear(x, e_param[126], e_param[127])

        return x



