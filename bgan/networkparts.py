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

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
        
    def forward(self, input):
        return self.func(input)        
        
class BadGanG(nn.Module):
    
    def __init__(self, d=128, z_dim=100):
        super(BadGanG, self).__init__()
        self.z_dim = z_dim
        self.core_net = nn.Sequential(
            nn.Linear(z_dim, 4*4*(4*d)), nn.BatchNorm1d(4*4*(4*d)), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0),4*d,4,4)),
            nn.ConvTranspose2d(4*d,2*d,5,2,2,1), nn.BatchNorm2d(2*d), nn.ReLU(),
            nn.ConvTranspose2d(2*d,  d,5,2,2,1), nn.BatchNorm2d(d),   nn.ReLU(),
            nn.ConvTranspose2d(d  ,  3,5,2,2,1), nn.Tanh(),
        )
    def forward(self, z):
        return self.core_net(z)
    
class BadGanD(nn.Module):
    
    def __init__(self, d=64, numClasses=2):
        super(BadGanD, self).__init__()
        self.numClasses = numClasses
        self.core_net = nn.Sequential(
            nn.Conv2d(  3,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.2),
            nn.Conv2d(  d,  d,3,1,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.2),
            nn.Conv2d(  d,  d,3,2,1), nn.BatchNorm2d(  d), nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.Conv2d(  d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,2*d,3,1,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,2*d,3,2,1), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.Conv2d(2*d,2*d,3,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,2*d,1,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            nn.Conv2d(2*d,2*d,1,1,0), nn.BatchNorm2d(2*d), nn.LeakyReLU(0.2),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
            nn.Linear(2*d,self.numClasses),
        ) # todo: replace batchnorm with weightnorm
    def forward(self, x):
        return self.core_net(x)