from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.autograd import Variable

def get_posterior(model, test_loader, magnitude, temperature, stdv):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    output = []

    for j, data in enumerate(test_loader):
        data, _ = data
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        outputs,gx = model(data)
        # outputs = model(data)

        labels = torch.max(outputs,1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()

        gradient = torch.sign(data.grad.data)

        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / stdv[0])
        
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / stdv[1])
        
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / stdv[2])
        
        tempInputs = torch.add(data.data, -magnitude, gradient)
        outputs,gx = model(Variable(tempInputs, volatile=True))
        # outputs = model(Variable(tempInputs, volatile=True))
        hx = outputs*gx

        # soft_out = F.softmax(outputs, dim=1)
        # soft_out, _ = torch.max(outputs.data, dim=1)
         
        max_out = torch.max(hx.data, dim=1)[0]
        output.append(max_out.cpu())
        # output.append(soft_out.cpu())

    output = np.concatenate(output)
    return output
