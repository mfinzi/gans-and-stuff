import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import tensorboardX
from torch.utils.data import DataLoader
from bgan.utils import to_var_gpu, prettyPrintLog
from bgan.dataloaders import getUandLloaders
import torch.nn.functional as F
import itertools
import shutil
import copy

class CnnTrainer:
    
    def __init__(self, CNN, datasets, save_dir, lr=2e-4, lab_BS=32, ul_BS=32,
                amntLab=1):
        shutil.rmtree(save_dir, ignore_errors=True) # Make clean
        self.writer = tensorboardX.SummaryWriter(save_dir)

        assert torch.cuda.is_available(), "CUDA or GPU not working"
        self.CNN = CNN.cuda()
        self.writer.add_text('ModelSpec','CNN: '+type(CNN).__name__)

        # Init hyperparameter dictionary
        self.hypers = {'lr':lr, 'amntLab':amntLab, 'lab_BS':lab_BS, 'ul_BS':ul_BS}
        self.optimizer = optim.Adam(self.CNN.parameters(),lr=lr)
        self.lab_train, self.dev, self.test, self.unl_train \
                    = self.getDataLoaders(datasets,lab_BS,ul_BS,amntLab)

    def getDataLoaders(self, datasets, lab_BS, ul_BS, amntLab):
        trainset, devset, testset = datasets
        unl_loader, lab_loader = getUandLloaders(trainset,amntLab,lab_BS,ul_BS,num_workers=0)
        dev_loader = DataLoader(devset, batch_size = 64, num_workers = 0)
        test_loader = DataLoader(testset, batch_size = 64, num_workers = 0)
        return lab_loader, dev_loader, test_loader, unl_loader

    def loss(self, x, y):
        # Losses for labeled samples are -log(P(y=yi|x))
        loss = nn.CrossEntropyLoss()(self.CNN(x),y)
        return loss

    def step(self, *data):
        varData = tuple(map(to_var_gpu, data))
        self.optimizer.zero_grad()
        loss = self.loss(*varData)
        loss.backward()
        self.optimizer.step()
        return loss
         
    def batchAccuracy(self, x, y):
        x, y = to_var_gpu(x),  to_var_gpu(y)
        self.CNN.eval()
        predictions = self.CNN(x).max(1)[1].type_as(y)
        correct = predictions.eq(y).cpu().data.numpy().mean()
        self.CNN.train()
        return correct
    
    def getDevsetAccuracy(self):
        accSum = 0
        for xy in self.dev:
            accSum += self.batchAccuracy(*xy)
        acc = accSum / len(self.dev)
        self.writer.add_text('metrics','Devset_Accuracy: = %.3f'%acc)
        return acc

    def writeHypers(self):
        for tag, value in self.hypers.items():
            self.writer.add_text('ModelSpec',tag+' = '+str(value))

    def train(self, numEpochs=100):
        self.writeHypers()
        numBatchesPerEpoch = len(self.lab_train)
        numSteps = numEpochs*numBatchesPerEpoch
        lab_iter = iter(self.lab_train)
        cycle_dev_iter = itertools.cycle(self.dev)
        # Get the input data and run optimizer step
        for step in range(numSteps):
            x, y = next(lab_iter)
            batchLoss = self.step(x, y)
            if step%100==0:
                # Add logging data to tensorboard writer
                logData = {
                    'Training_Loss':batchLoss.cpu().data[0],
                    'Train_Acc':self.batchAccuracy(x, y),
                    'Val_Acc':self.batchAccuracy(*next(cycle_dev_iter)),}
                self.writer.add_scalars('metrics', logData, step)
            if step%1000==0:
                epoch = step / numBatchesPerEpoch
                prettyPrintLog(logData, epoch, numEpochs, step, numSteps)
        print('Devset Acc: %.3f'%self.getDevsetAccuracy())

    # def initSWA(self):
    #     self.n = 0
    #     self.SWA = copy.deepcopy(self.CNN)
    
    # def updateSWA(self):
    #     n = self.n
    #     for param1, param2 in zip(self.SWA.parameters(),self.CNN.parameters()):
    #         param1.data = param1.data*n/(n+1) + param2.data/(n+1)
    #     self.n+=1
    #     # # Keep Running average of the weights, updated every epoch
    #     #         if self.hypers['swa']:
    #     #             self.updateSWA()
    # if swa: self.initSWA()