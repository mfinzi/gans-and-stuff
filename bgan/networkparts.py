import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# weight init is automatically done in the module initialization
# see https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
def weight_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# custom weights initialization called on netG and netD
def weights_init_DC(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


class BadGanG(nn.Module):
    
    def __init__(self, d=128, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.core_net = nn.Sequential(
            nn.Linear(z_dim, 4*4*(4*d)), nn.BatchNorm1d(4*4*(4*d)), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0),4*d,4,4)),
            nn.ConvTranspose2d(4*d,2*d,5,2,2,1), nn.BatchNorm2d(2*d), nn.ReLU(),
            nn.ConvTranspose2d(2*d,  d,5,2,2,1), nn.BatchNorm2d(d),   nn.ReLU(),
            nn.ConvTranspose2d(d  ,  3,5,2,2,1),#, train_scale=True, init_stdv=.1),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.core_net(z)

class BadGanGv2(nn.Module):
    
    def __init__(self, d=128, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.core_net = nn.Sequential(
            nn.Linear(z_dim, 4*4*(4*d)), nn.BatchNorm1d(4*4*(4*d)), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0),4*d,4,4)),
            nn.ConvTranspose2d(4*d,2*d,5,2,2,1), nn.BatchNorm2d(2*d), nn.ReLU(),
            nn.ConvTranspose2d(2*d,  d,5,2,2,1), nn.BatchNorm2d(d),   nn.ReLU(),
            nn.ConvTranspose2d(d  ,  3,5,2,2,1),#, train_scale=True, init_stdv=.1),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.core_net(z)
    


class BadGanDbn(nn.Module):
    
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.feature_net = nn.Sequential(
            nn.Conv2d(  3,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1), 
            nn.Conv2d(  d,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Conv2d(  d,  d,3,2,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(  d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,2,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(2*d,2*d,3,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.core_net = nn.Sequential(
            self.feature_net,
            nn.Linear(2*d,self.numClasses),
        )
        
    def forward(self, x, getFeatureVec=False):
        if getFeatureVec: return self.feature_net(x)
        else: return self.core_net(x)

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, inpt):
        if self.training:
            noise = Variable(inpt.data.new(inpt.size()).normal_(std=self.sigma))
            return inpt + noise
        else:
            return inpt

class BadGanDbn2(nn.Module):
    
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.feature_net = nn.Sequential(
            GaussianNoise(0.15),
            nn.Conv2d(  3,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1), 
            nn.Conv2d(  d,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Conv2d(  d,  d,3,2,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(  d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,2,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(2*d,4*d,3,1,0), nn.BatchNorm2d(4*d), nn.LeakyReLU(0.1),
            nn.Conv2d(4*d,2*d,1,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,  d,1,1,0), nn.BatchNorm2d(  d), nn.LeakyReLU(0.1),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.core_net = nn.Sequential(
            self.feature_net,
            nn.Linear(d,self.numClasses),
        )
        
    def forward(self, x, getFeatureVec=False):
        if getFeatureVec: return self.feature_net(x)
        else: return self.core_net(x)

class BadGanDin(nn.Module):
    
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.feature_net = nn.Sequential(
            nn.Conv2d(  3,  d,3,1,1), nn.InstanceNorm2d(  d), nn.LeakyReLU(0.1), 
            nn.Conv2d(  d,  d,3,1,1), nn.InstanceNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Conv2d(  d,  d,3,2,1), nn.InstanceNorm2d(  d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(  d,2*d,3,1,1), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,1,1), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,2,1), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(2*d,2*d,3,1,0), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.InstanceNorm2d(2*d), nn.LeakyReLU(0.1),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.core_net = nn.Sequential(
            self.feature_net,
            nn.Linear(2*d,self.numClasses),
        )
        
    def forward(self, x, getFeatureVec=False):
        if getFeatureVec: return self.feature_net(x)
        else: return self.core_net(x)

class BadGanDon(nn.Module):
    
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.feature_net = nn.Sequential(
            nn.Conv2d(  3,  d,3,1,1), nn.LeakyReLU(0.1), 
            nn.Conv2d(  d,  d,3,1,1), nn.LeakyReLU(0.1),
            nn.Conv2d(  d,  d,3,2,1), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(  d,2*d,3,1,1), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,1,1), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,3,2,1), nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Conv2d(2*d,2*d,3,1,0), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.LeakyReLU(0.1),
            nn.Conv2d(2*d,2*d,1,1,0), nn.LeakyReLU(0.1),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.core_net = nn.Sequential(
            self.feature_net,
            nn.Linear(2*d,self.numClasses),
        )
        
    def forward(self, x, getFeatureVec=False):
        if getFeatureVec: return self.feature_net(x)
        else: return self.core_net(x)



class DCganG(nn.Module):
    def __init__(self, d=64, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.core_net = nn.Sequential(
            Expression(lambda tensor: tensor.unsqueeze(2).unsqueeze(3)),
            nn.ConvTranspose2d(z_dim,4*d,4,1,0), nn.BatchNorm2d(4*d), nn.ReLU(),
            nn.ConvTranspose2d(  4*d,2*d,4,2,1), nn.BatchNorm2d(2*d), nn.ReLU(),
            nn.ConvTranspose2d(  2*d,  d,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(    d,  3,4,2,1), nn.Tanh(),
        )
    def forward(self, z):
        return self.core_net(z)


class DCganD(nn.Module):
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.core_net = nn.Sequential(
            nn.Conv2d(  3,  d,4,2,1),                      nn.LeakyReLU(0.2),
            nn.Conv2d(  d,2*d,4,2,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,4*d,4,2,1), nn.BatchNorm2d(4*d), nn.LeakyReLU(0.2),
            nn.Conv2d(4*d,self.numClasses,4,1,0),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
    def forward(self, x):
        return self.core_net(x)

class DCganDv2(nn.Module):
    def __init__(self, d=64, numClasses=2):
        super().__init__()
        self.numClasses = numClasses
        self.feature_net = nn.Sequential(
            nn.Conv2d(  3,  d,4,2,1),                      nn.LeakyReLU(0.2),
            nn.Conv2d(  d,2*d,4,2,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,4*d,4,2,1), nn.BatchNorm2d(4*d), nn.LeakyReLU(0.2),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.core_net = nn.Sequential(
            self.feature_net,
            nn.Linear(4*d,self.numClasses),
        )
    def forward(self, x, getFeatureVec=False):
        if getFeatureVec: return self.feature_net(x)
        else: return self.core_net(x)




# cross entropy loss manual
# batch_size = x.size()[0]
# batchIndices = torch.arange(0,batch_size).type_as(y.data)
# # logSoftMax = nn.LogSoftmax(dim=1)
# lab_losses = -1*logSoftMax(self.CNN(x))[batchIndices,y]
# loss = torch.mean(lab_losses)