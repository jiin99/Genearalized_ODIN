from __future__ import print_function
import json
import utils
import argparse
import torch
import metrics
import torch.backends.cudnn as cudnn
import os
import torchvision
from densenet import DenseNet3_C,init_normal
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch code: ODIN')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
parser.add_argument('--save_root', required=True, help='log save path')
parser.add_argument('--aug_crop', default='F', help='aug_crop T = crop')
parser.add_argument('--network', type=str, default='dense',  required=True, help='dense | res | vgg16')
parser.add_argument('--pre_trained_net', default='ckpt', help='names')
parser.add_argument('--gpu', type=str, default='3', help='gpu index')
args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    # Make folder and save path
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    with open('{0}/valid_configuration.json'.format(args.save_root), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    #cifar10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    #cifar 100
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    #     ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    validset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    total_len = len(testset)
    indices = list(range(total_len))
    #validation : 0.1 test : 0.9
    split = int(np.floor(0.9*total_len))

    valid_idx, test_idx = indices[split:], indices[:split]
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    in_validloader = torch.utils.data.DataLoader(
        testset, batch_size=64, sampler = valid_sampler,shuffle=False, num_workers=2)
    in_testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, sampler = test_sampler,shuffle=False, num_workers=2)

    # load networks
    if args.network == 'dense':
        net = DenseNet3_C(depth=100, num_classes=10, growth_rate=12,
                          reduction=0.5, bottleneck=True,dropRate=0.0).cuda()
    elif args.network == 'res':
        model = resnet.resnet110(**network_dict).cuda()
    elif args.network == 'vgg16':
        model = vgg.vgg16(**network_dict).cuda()
    
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    # set the path to pre-trained model and output
    pre_trained_net = '/daintlab/home/jiin9/ood/class/checkpoint_div_C_10/' + args.pre_trained_net + '.pt'
    net.load_state_dict(torch.load(pre_trained_net))

    print('load model: ' + args.network)

    magnitude_list = [0.0025,0.005,0.01,0.02,0.04,0.08]

    # Baseline
    in_output = utils.get_posterior(net, in_validloader, 0, 1, [0.267, 0.256, 0.276])
    
    di = {}
    for magnitude in magnitude_list:
        in_output = utils.get_posterior(net, in_validloader, magnitude, 1, [0.267, 0.256, 0.276])
        di[magnitude] = sum(in_output)
    print(di)

if __name__ == '__main__':
    main()