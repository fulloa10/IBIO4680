#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:23:34 2018

@author: f.ulloa10
"""

#Laboratorio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import tqdm
import os

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, kernel_size=3,padding=(1,1)) #Channels input: 1, c output: 63, filter of size 3
        self.conv2 = nn.Conv2d(192, 300, kernel_size=3,padding=(1,1))
        self.conv3 = nn.Conv2d(300, 400, kernel_size=3,padding=(1,1))
        self.conv1_bn = nn.BatchNorm2d(192)
        self.conv2_bn = nn.BatchNorm2d(300)
        self.conv3_bn = nn.BatchNorm2d(400)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(400*64, 100)
        self.fc2 = nn.Linear(100, 25)
        self.soft=nn.Softmax()
        
    def forward(self, x, verbose=False):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, 4)
        x = x.view(-1, 400*64)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.fc2(x)
        x=self.soft(x)
        return x

    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.CrossEntropyLoss()
        
def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.3,))])
    data_train = datasets.ImageFolder('texturesCNN/train_128/',transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    data_eval = datasets.ImageFolder('texturesCNN/val_128/',transform_train)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=batch_size, shuffle=True)
    return train_loader, eval_loader


def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.cuda(); data = Variable(data)
        target = target.cuda(); target = Variable(target)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
        
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    
    

def test(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.cuda(); data = Variable(data, volatile=True)
        target = target.cuda(); target = Variable(target, volatile=True)
        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    return Acc

    
def evalu(data_loader, model):
    model.eval()
    res=np.array([])
    for (data,target) in data_loader :
        data = data.cuda(); data = Variable(data, volatile=True)
        output = model(data)
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        arg_max_out=arg_max_out.numpy()
        res=np.append(res,arg_max_out)
        print(arg_max_out)
    return res

def loadModel(model, modelPath):
    model.load_state_dict(torch.load(modelPath))
    

if __name__=='__main__':
    epochs=30
    batch_size=32
    TEST=True
    train_loader, test_loader = get_data(batch_size)

    model = Net()
    model.cuda()

    model.training_params()
    loadModel(model,'CNN1.pth')
    model.cuda()
    print_network(model, 'Conv network')
