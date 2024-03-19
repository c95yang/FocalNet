import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval
import numpy as np
import random
from ptflops import get_model_complexity_info
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(result_folder):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(result_folder + args.model_name + '/'):
        os.makedirs(result_folder + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)

    #print(model)

    x_fake = torch.randn(3, 256, 256)
    macs_ssm = model.flops(x_fake) / 1e9

    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"Parameter {name} is on device: {param.device}")

    macs, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True, verbose=True)

    print(f"Model Parameters: {params}")

    macs = macs.replace('GMac', '')
    macs_float = float(macs)
    print(f"Model FLOPs: {macs} GFLOPs, VSSG FLOPs: {macs_ssm:.2f} GFLOPs. Total: {macs_float + macs_ssm:.2f} GFLOPs")
        
    model.cuda()
    if args.mode == 'train':
        _train(model, args)
        #print(f"SSM FLOPs: {g['value']}")
        #print(f"SSM Parameters: {p['value']}")

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    result_folder = 'results/'
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus'], type=str)
#    parser.add_argument('--data_dir', type=str, default='/data/ir_datasets/reside-indoor')
#    parser.add_argument('--data_dir', type=str, default='/mnt/nvme0n1/cyn/datasets/its/reside-indoor')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/home/cc/Documents/data/reside-indoor')
    # Train
    parser.add_argument('--batch_size', type=int, default=4)#4
    parser.add_argument('--learning_rate', type=float, default=1e-4)#1e4
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)#300
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    # parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # Test
    parser.add_argument('--test_model', type=str, default='/home/cc/Downloads/model.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join(result_folder, 'mean', 'full')
    args.result_dir = os.path.join(result_folder, args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/MIMOUNet.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
