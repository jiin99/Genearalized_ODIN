from __future__ import print_function
import json
import utils
import argparse
import torch
import metrics
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets
from torchvision import transforms
import os
import numpy as np
import lmdb
from densenet import init_normal, DenseNet3_baseline, DenseNet3_C,DenseNet3_I
from torch.utils.data.sampler import SubsetRandomSampler

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
    transform_tiny = transforms.Compose([
        transforms.RandomCrop(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
    ])
    transform_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
    ])
    transform_lsun = transform = transforms.Compose([
        transforms.RandomCrop(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    validset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    test_out_svhn_set = torchvision.datasets.SVHN(
         root='./data', split = 'test', download=True, transform=transform_svhn)
    test_out_tiny_set = datasets.ImageFolder('/daintlab/home/jiin9/ood/ood/data/tiny_img/tiny_images',transform = transform_tiny)
    # test_out_lsun_set = torchvision.datasets.LSUN(
    #      root='./data', classes = 'test', transform=transform_lsun)
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
    out_svhn_testloader = torch.utils.data.DataLoader(
        test_out_svhn_set, batch_size=64, shuffle = False, num_workers=2)
    # out_lsun_testloader = torch.utils.data.DataLoader(
    #     test_out_lsun_set, batch_size=64, shuffle = False, num_workers=2)
   
    #num_classes = 10
    
    if args.network == 'dense':
        net = DenseNet3_C(depth=100, num_classes = 10, growth_rate=12, reduction=0.5, bottleneck=True,dropRate=0.0).cuda()
        
    # set the path to pre-trained model and output
    pre_trained_net = '/daintlab/home/jiin9/ood/class/checkpoint_div_C_10_again/' + args.pre_trained_net + '.pt'
    net.load_state_dict(torch.load(pre_trained_net))

    #cifar10
    in_score = utils.get_posterior(net, in_testloader, 0.0025, 1, [0.247, 0.243, 0.262])
    ood_svhn_score = utils.get_posterior(net, out_svhn_testloader, 0.0025, 1,  [0.247, 0.243, 0.262])
   
    
    # cifar100
    # in_score = utils.get_posterior(net, in_testloader, 0.0, 1, [0.267, 0.256, 0.276])
    # ood_svhn_score = utils.get_posterior(net, out_svhn_testloader, 0.0, 1,  [0.267, 0.256, 0.276])


    '''in_labels = np.ones(len(in_score))
    ood_svhn_labels = np.zeros(len(ood_svhn_score))
    ood_lsun_labels = np.zeros(len(ood_svhn_score))
    print('svhn len : {0}, lsun len : {1}'.format(len(ood_svhn_score),len(ood_lsum_score)))
    # ood_scores.shape[0]
    labels_svhn = np.concatenate([in_labels, ood_svhn_labels])
    scores_svhn = np.concatenate([in_score, ood_svhn_score])

    print('----------out : svhn-------------')
    auroc_score = metrics.auroc(labels_svhn, scores_svhn, 'cifar10')
    tnr_tpr5_score = metrics.tnr_tpr95(labels_svhn, scores_svhn, 'cifar10')
    metrics.met(labels_svhn,scores_svhn,'cifar10')

    labels_lsun = np.concatenate([in_labels, ood_lsun_labels])
    scores_lsun = np.concatenate([in_score, ood_lsun_score])

    print('--------------out : lsun---------')
    auroc_score = metrics.auroc(labels_lsun, scores_lsun, 'cifar10')
    tnr_tpr5_score = metrics.tnr_tpr95(labels_lsun, scores_lsun, 'cifar10')
    metrics.met(labels_lsun,scores_lsun,'cifar10')
'''

    in_labels = np.ones(len(in_score))
    ood_svhn_labels = np.zeros(len(ood_svhn_score))
    labels_svhn = np.concatenate([in_labels, ood_svhn_labels])
    scores_svhn = np.concatenate([in_score, ood_svhn_score])

    auroc_score = metrics.auroc(labels_svhn, scores_svhn, 'cifar10')
    tnr_tpr5_score = metrics.tnr_tpr95(labels_svhn, scores_svhn, 'cifar10')
    metrics.met(labels_svhn,scores_svhn,'cifar10')





if __name__ == '__main__':
    main()
