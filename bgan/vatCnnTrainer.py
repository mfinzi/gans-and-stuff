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

class VatCnnTrainer(CnnTrainer):
    def __init__(self, *args, regScale=1, advEps=2, entMin=True,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'regScale':regScale, 'advEps':advEps, 
                            'entMin':entMin})

    def loss(self, x_lab, y_lab, x_unlab):
        loss = nn.CrossEntropyLoss()(self.CNN(x_lab),y_lab)
        eps = self.hypers['advEps']; entMin = self.hypers['entMin']
        unlabLDS = LDSloss(self.CNN, x_unlab, eps, entMin)
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

            if step%200==0:
                x_val, y_val = next(cycle_dev_iter)
                #xv,yv = to_var_gpu(x_lab), to_var_gpu(y_lab)
                #train_crossent = (nn.CrossEntropyLoss()(self.CNN(xv),yv)).cpu().data[0]
                # Add logging data to tensorboard writer
                logData = { 'Train_loss': train_loss.cpu().data[0],
                            'Train_Acc':self.batchAccuracy(x_lab, y_lab),
                            'Val_Acc':self.batchAccuracy(x_val,y_val),
                            'Train_LDS':LDSloss(self.CNN, to_var_gpu(x_lab),
                                         self.hypers['advEps']).cpu().data[0],
                            #'tcross':train_crossent
                            }
                self.writer.add_scalars('metrics', logData, step)
            
            if step%2000==0:
                # Print the logdata and write out adversarial images
                epoch = step / numBatchesPerEpoch
                prettyPrintLog(logData, epoch, numEpochs, step, numSteps)
                someX = to_var_gpu(x_unlab[:16])
                r_adv = self.hypers['advEps'] * getAdvPert(self.CNN, someX)
                adversarialImages = (someX + r_adv).cpu().data
                imgComparison = torch.cat((adversarialImages, x_unlab[:16]))
                self.writer.add_image('adversarialInputs', #,range=(-2.5,2.5)
                    vutils.make_grid(imgComparison,normalize=True), step)
        print('Devset Acc: %.3f'%self.getDevsetAccuracy())