import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torchvision.utils as vutils
from bgan.cnnTrainer import CnnTrainer
from bgan.vatLDS import LDSloss, getAdvPert
from bgan.utils import to_var_gpu, prettyPrintLog
import itertools

class MeanTrainer(CnnTrainer):
    def __init__(self, *args, regScale=1, EMA=0.999,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'regScale':regScale, 'EMA':EMA})

        self.teacher = copy.deepcopy(self.CNN)
        for param in self.teacher.parameters():
            param.detach_()

    def updateTeacherWeights(self, step):
        # Use the true average until the exponential average is more correct
        alpha = min(1-1/(step +1), alpha)
        paramIter = zip(self.teacher.parameters(), self.CNN.parameters())
        for teacherParam, studentParam in paramIter:
            teacherParam.data.mul_(alpha).add_(1-alpha, studentParam.data)


    def loss(self, x_lab, y_lab, x_unlab):
        unlabLDS = LDSloss(self.CNN, x_unlab, eps, entMin)
        loss = nn.CrossEntropyLoss()(self.CNN(x_lab),y_lab)
        loss += self.hypers['regScale']*unlabLDS
        return loss

    def train(self, numEpochs = 1200):
        self.writeHypers()
        cycle_dev_iter = itertools.cycle(self.dev)
        unl_iter, lab_iter = iter(self.unl_train), iter(self.lab_train)
        numBatchesPerEpoch = len(self.lab_train)
        numSteps = numEpochs*numBatchesPerEpoch

        for step in range(numSteps):
            # Get the input data and run optimizer step
            x_unlab, _ = next(unl_iter)
            x_lab, y_lab = next(lab_iter)
            train_loss = self.step(x_lab, y_lab, x_unlab)
            self.updateTeacherWeights(step)

            if step%100==0:
                # Add logging data to tensorboard writer
                logData = { 'Train_loss': train_loss.cpu().data[0],
                            'Train_Acc':self.batchAccuracy(x_lab, y_lab),
                            'Val_Acc':self.batchAccuracy(*next(cycle_dev_iter)),
                            }
                self.writer.add_scalars('metrics', logData, step)
            
            if step%2000==0:
                # Print the logdata and write out adversarial images
                epoch = step / numBatchesPerEpoch
                prettyPrintLog(logData, epoch, numEpochs, step, numSteps)
        print('Devset Acc: %.3f'%self.getDevsetAccuracy())