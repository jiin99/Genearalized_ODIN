from densenet import init_normal, DenseNet3_C, DenseNet3_baseline
from util import Logger, AverageMeter
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import time
import argparse
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import wandb
import json
wandb.init(project="pro0402")
is_best = 0
'''device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True'''



def main():
    global is_best
    train_logger = Logger(os.path.join('./loss', 'train_logger_10_div_C_again.log')) 
    valid_logger = Logger(os.path.join('./loss', 'valid_logger_10_div_C_again.log'))
    gx_li = []
    v_gx_li = [] 
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
   
    #cifar10
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    '''#cifar 100
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])'''

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4)
    
    
    num_classes = 10

    print('builing model')
    model = DenseNet3_C(100,num_classes).apply(init_normal).cuda()
    #Set optimizer
    
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0001)#,weight_decay = 0.96)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,225], gamma=0.1)
    #Set criterion
    criterion = nn.CrossEntropyLoss().cuda()
   
    #Start Train
    for epoch in tqdm(range(0,300)):
        #train
        train(model,trainloader,64,optimizer,epoch,criterion,train_logger,gx_li)
        acc = validate(model,testloader,64,optimizer,epoch,criterion,valid_logger,v_gx_li)
        
        #torch.save(model.state_dict(), 'single_module_cnn/each_epoch{0}.pt'.format(epoch))

        if acc > is_best:
            if not os.path.isdir('checkpoint_div_C_10_again'):
                os.mkdir('checkpoint_div_C_10_again')
            torch.save(model.state_dict(), './checkpoint_div_C_10_again/ckpt_best{0}.pt'.format(epoch))
            is_best = acc
        torch.save(train_logger,'loss/checkpoint_div_C_10_train_again_logger.pt')
        torch.save(valid_logger,'loss/checkpoint_div_C_10_valid_again_logger.pt')
        torch.save(gx_li,'loss/checkpoint_div_C_100_again_gx.pt')
        torch.save(v_gx_li,'loss/checkpoint_div_C_10_again_gx_li.pt')
        scheduler.step()
    print("Done") 

def train(model,train_loader,batch_size,optimizer,epoch,criterion,train_logger,gx_li):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy,train_loss = AverageMeter(), AverageMeter()
    correct = 0
    model.train()
    end = time.time() 
    for step,(data, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.cuda(), target.cuda()
        output, gx = model(data)

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().float()
        acc = correct / (batch_size * (step + 1)) * 100
        accuracy.update(acc.item(), data.shape[0])
        
        loss = criterion(output,target)
        
        train_loss.update(loss.item(), data.shape[0])
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        wandb.log({'epoch': epoch, 'train_loss': train_loss.val})
        wandb.log({'epoch': epoch, 'train_loss_avg': train_loss.avg})
        wandb.log({'epoch': epoch, 'train_acc': accuracy.val})
        wandb.log({'epoch': epoch, 'train_acc_avg': accuracy.avg})
        wandb.log({'epoch': epoch, 'gx': gx.detach().cpu().numpy()})
        if step % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=train_loss, acc = accuracy))
            wandb.log({'epoch': epoch, 'tr_loss':train_loss.avg})
            wandb.log({'epoch': epoch, 'tr_acc':accuracy.avg})
        gx_li.append(gx)
    #li.append(tr_loss/len(trn_dl))
    train_logger.write([epoch, train_loss.avg, accuracy.avg])

    #torch.save(model.state_dict(), 'weight/state_dict9.pt')


def validate(model,val_loader,batch_size,optimizer,epoch,criterion,val_logger,gx_li):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_accuracy,val_loss = AverageMeter(),AverageMeter()
    correct = 0
    #switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        
        for i, (data,target) in tqdm(enumerate(val_loader)):
            data, target = data.cuda(), target.cuda()
            output, gx = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().float()
            acc = correct / (batch_size * (i + 1)) * 100
            val_accuracy.update(acc.item(), data.shape[0])
            loss = criterion(output,target)
            val_loss.update(loss.item(),data.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()


            gx_li.append(gx)
            
            wandb.log({'epoch': epoch, 'v_gx':gx.detach().cpu().numpy()})
            wandb.log({'epoch': epoch, 'v_loss_avg':val_loss.avg})
            wandb.log({'epoch': epoch, 'v_acc_avg':val_accuracy.avg})

            if i % 100 == 0:
                print('[{0}/{1}]\t'
                      'V_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'V_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'V_acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                                i, len(val_loader), batch_time=batch_time, loss=val_loss, acc=val_accuracy))
                wandb.log({'epoch': epoch, 'valval_loss': val_loss.val})
                wandb.log({'epoch': epoch, 'valval_loss_avg': val_loss.avg})
                wandb.log({'epoch': epoch, 'valval_acc': val_accuracy.val})
                wandb.log({'epoch': epoch, 'valval_acc_avg': val_accuracy.avg})
        
        
        val_logger.write([epoch, val_loss.avg, val_accuracy.avg])
    return val_accuracy.avg
if __name__ == "__main__":
    main()