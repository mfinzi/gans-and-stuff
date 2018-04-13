import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import tensorboardX
from torch.utils.data import DataLoader
from bgan.utils import to_var_gpu, to_lambda, prettyPrintLog
from bgan.dataloaders import getUandLloaders, getLoadersBalanced
#from bgan.schedules import CosineAnnealer, AdamR

import torch.nn.functional as F
import itertools
import shutil
import copy

class CnnTrainer:
    
    def __init__(self, CNN, datasets, save_dir, 
                base_lr=2e-4, lab_BS=32, ul_BS=32,
                amntLab=1, num_workers=2, opt_constr=None,
                extraInit=lambda:None, lr_lambda = lambda e: 1):
                #cycle_length=100, cycle_mult=1.3, with_restart=True):

        # Setup tensorboard logger
        #shutil.rmtree(save_dir, ignore_errors=True) # Make clean
        self.writer = tensorboardX.SummaryWriter(save_dir)
        self.metricLog = {}
        self.scheduleLog = {}

        # Setup Network and Optimizers
        assert torch.cuda.is_available(), "CUDA or GPU not working"
        self.CNN = CNN.cuda()
        self.writer.add_text('ModelSpec','CNN: '+type(CNN).__name__)
        if opt_constr is None: opt_constr = optim.Adam
        self.optimizer = opt_constr(self.CNN.parameters(), base_lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)

        # Setup Dataloaders and Iterators
        self.lab_train, self.dev, self.test, self.unl_train \
                    = self.getDataLoaders(datasets,lab_BS,ul_BS,amntLab, num_workers)
        self.train_iter = iter(self.lab_train)
        self.numBatchesPerEpoch = len(self.lab_train)

        # Init hyperparameter dictionary
        self.hypers = {'base_lr':base_lr, 'amntLab':amntLab, 'lab_BS':lab_BS,
                         'ul_BS':ul_BS}
        # Extra work to do (used for subclass)
        extraInit()
        # Log the hyper parameters
        for tag, value in self.hypers.items():
            self.writer.add_text('ModelSpec',tag+' = '+str(value))

    def getDataLoaders(self, datasets, lab_BS, ul_BS, amntLab, num_workers):
        """ Handles getting cyclic dataloaders (ul and lab) for data """
        trainset, devset, testset = datasets
        unl_loader, lab_loader = getLoadersBalanced(trainset,amntLab,lab_BS,ul_BS,
                                            num_workers=num_workers,pin_memory=True)
        dev_loader = DataLoader(devset, batch_size = 64, num_workers = num_workers)
        test_loader = DataLoader(testset, batch_size = 64, num_workers = num_workers)
        return lab_loader, dev_loader, test_loader, unl_loader

    def train(self, numEpochs=100):
        """ The main training loop called (also for subclasses)"""
        for epoch in range(numEpochs):
            self.lr_scheduler.step(epoch); self.epoch = epoch
            for i in range(self.numBatchesPerEpoch):
                trainData = to_var_gpu(next(self.train_iter))
                self.step(*trainData)
                self.logStuff(i, epoch, numEpochs, trainData)

    def step(self, *data):
        self.optimizer.zero_grad()
        loss = self.loss(*data)
        loss.backward()
        self.optimizer.step()

    def loss(self, *data):
        """ Basic cross entropy loss { -E[log(P(y=yi|x))] } """
        x, y = data
        loss = nn.CrossEntropyLoss()(self.CNN(x),y)
        return loss

    def logStuff(self, i, epoch, numEpochs, trainData):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i + epoch*self.numBatchesPerEpoch
        numSteps = numEpochs*self.numBatchesPerEpoch
        if step%2000==0:
            xy_labeled = self.getLabeledXYonly(trainData)
            self.metricLog['Train_Acc(Batch)'] = self.batchAccuracy(*xy_labeled)
            self.metricLog['Val_Acc'] = self.getDevsetAccuracy()
            self.writer.add_scalars('metrics', self.metricLog, step)
            prettyPrintLog(self.metricLog, epoch, numEpochs, step, numSteps)

            self.scheduleLog['lr'] = self.lr_scheduler.get_lr()[0]
            self.writer.add_scalars('schedules', self.scheduleLog, step)

    def getLabeledXYonly(self, trainData):
        """ should return a tuple (x,y) that will be used to calc acc """
        return trainData

    def batchAccuracy(self, *labeledData):
        x, y = labeledData
        self.CNN.eval()
        predictions = self.CNN(x).max(1)[1].type_as(y)
        correct = predictions.eq(y).cpu().data.numpy().mean()
        self.CNN.train()
        return correct
    
    def getDevsetAccuracy(self):
        accSum = 0
        for xy in self.dev:
            xy = to_var_gpu(xy)
            accSum += self.batchAccuracy(*xy)
        acc = accSum / len(self.dev)
        return acc


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