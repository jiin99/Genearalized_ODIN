import os
import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from collections import Iterable
'''
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count'''

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass
            
class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], correct.squeeze()
'''
def save_checkpoint(state, is_best, save_path):
    filepath = os.path.join(save_path, 'checkpoint.pth')
    torch.save(state, filepath)
    print("Saving Model")
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))
'''
'''def save_checkpoint(state, is_best,i):
    torch.save(state, f'weight/checkpoint{i}.pth.tar')
    print("Saving Model")
    if is_best:
        shutil.copyfile(f'weight/checkpoint{i}.pth.tar', f'weight/model_best{i}.pth.tar')

def save_data(name, data, save_path):
    df_data = pd.DataFrame(data)
    df_data.to_csv(save_path + '{0}_data.csv'.format(name), encoding='ms949')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model'''

def save_checkpoint(state, is_best,i):
    torch.save(state, 'weight/checkpoint{}.pth'.format(i))
    print("Saving Model")
    if is_best:
        shutil.copyfile('weight/checkpoint{}.pth'.format(i), 'weight/model_best{}.pth'.format(i))

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer_state_dict']
    epochs = checkpoint['epoch']
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model