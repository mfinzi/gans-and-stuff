import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as tutils
import os
import itertools

class CnnTrainer:
    
    def __init__(self, CNN, dataloaders, save_dir, eta=2e-4):

        self.CNN = CNN.cuda()

        (self.lab_train, self.dev, self.test) = dataloaders

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.eta = eta
        self.optimizer = optim.Adam(self.CNN.parameters(),lr=self.eta)


    def loss(self, x, y):
        # batch_size = x.size()[0]
        # batchIndices = torch.arange(0,batch_size).type_as(y.data)
        # logSoftMax = nn.LogSoftmax(dim=1)
        # lab_losses = -1*logSoftMax(self.CNN(x))[batchIndices,y]
        # loss = torch.mean(lab_losses)
        
        # Losses for labeled samples are -log(P(y=yi|x))
        loss = nn.CrossEntropyLoss()(self.CNN(x),y)
        return loss
    

    def step(self, x, y):

        self.optimizer.zero_grad()
        l = self.loss(x,y)
        l.backward()
        self.optimizer.step()
         
    def batchAccuracy(self, x, y):
        predictions = self.CNN(x).max(1)[1].type_as(y)
        correct = predictions.eq(y).cpu().data.numpy().mean()
        return correct

    def train(self, epochs=100):
        datalog = np.empty((0,3)) # losses = [Dloss,Gloss]
        deviter = itertools.cycle(iter(self.dev))

        for epoch in range(epochs):
            for i, data in enumerate(self.lab_train):
                x = Variable(data[0].cuda())
                y = Variable(data[1].cuda().long().squeeze())
                self.step(x,y)

                if i%100==0:
                    batchTrainingLoss = self.loss(x,y).cpu().data.numpy().mean()
                    batchTrainAcc = self.batchAccuracy(x,y)
                    
                    xdev, ydev = next(deviter)
                    xdev, ydev = Variable(xdev.cuda()), Variable(ydev.cuda().long().squeeze())
                    batchDevAcc = self.batchAccuracy(xdev,ydev)
                    datalog = np.concatenate((datalog,np.array([batchTrainingLoss,batchTrainAcc,batchDevAcc])[None,:]))

            if epoch%1==0:
                nowData = np.mean(datalog[-7:],axis=0)
                print("[%3d/%d][%4d/%d] Train Loss: %.4f    Train Acc: %.3f   Test Acc: %.3f"
                    %(epoch,epochs,i,len(self.lab_train),nowData[0],nowData[1],nowData[2]))

        # todo: abstract loss & accuracy into data logger
        # write evaluation method with network in (eval) mode (for dropout)  
                