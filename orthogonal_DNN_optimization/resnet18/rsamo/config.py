import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    #data input setting
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--batchsize',type=int,default=128)
    parser.add_argument('--batchsize_param', type=int, default=1) #8
    parser.add_argument('--adam_lr', type=float, default =0.001)
    parser.add_argument('--epoch', type= int, default = 12000)
    parser.add_argument('--inner_epoch', type=int, default= 4)
    parser.add_argument('--lr_step', type = int, default= 6000)

    # meta_op
    parser.add_argument('--decay_rate', type= float, default= 0.1)
    parser.add_argument('--lstm_layers',type=int, default= 2)
    parser.add_argument('--hidden_size',type=int, default=20)
    parser.add_argument('--output_scale', type=float, default= 1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--p',type=int, default=10)
    parser.add_argument('--s_lr', type=float, default= 0.01)
    parser.add_argument('--e_lr', type=float, default = 0.01)
    parser.add_argument('--Neumann_alpha', type = float, default=0.1)
    parser.add_argument('--Neumann_series', type = float, default = 10)



    # hand_op
    parser.add_argument('--observation_epoch', type= int ,default = 0 )
    parser.add_argument('--hand_e_lr', type=float, default= 0.01)
    parser.add_argument('--hand_s_lr', type=float, default= 0.01)

    args = parser.parse_args()
    return args