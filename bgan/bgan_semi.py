import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as tutils
import os

def logOneMinusSoftmax(x):
    """ numerically more stable version of log(1-softmax(x)) """
    max_vals, _ = torch.max(x, 1)
    shifted_x = x - max_vals.unsqueeze(-1).expand_as(x)
    exp_x = torch.exp(shifted_x)
    sum_exp_x = exp_x.sum(1).unsqueeze(-1).expand_as(exp_x)
    first = torch.log(sum_exp_x - exp_x)
    second = torch.log(sum_exp_x)
    return first - second

class BGANS:
    
    def __init__(self, G, D, dataloaders, save_dir, MAP = True, 
                            eta=2e-4, alpha=.01):
        """ Creates a Bayesian GAN for generator G, discriminator D, and optimizer O.
        """
        self.G = G.cuda()
        self.D = D.cuda() #Assert self.D.K > 1

        (self.unl_train, self.lab_train, self.dev, self.test) = dataloaders
        self._setMode() # self.mod = {"uns","semi","fully"}

        self.MAP = MAP
        self.save_dir = save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        self.eta, self.alpha = eta, alpha
        self._initOptimizers()

    def _setMode(self):
        if self.unl_train!=None and self.lab_train!=None: self.mode = "semi"
        elif self.unl_train!=None: self.mode = "uns"
        elif self.lab_train!=None: self.mode = "fully"
        else: raise Exception("No viable dataloaders")
        

    def _initOptimizers(self):
        """
        Initializes the optimizers for BGAN.
        """
        if self.MAP:
            self.g_optimizer = optim.Adam(self.G.parameters(), 
                lr=self.eta, betas=(0.5, 0.999))
        else: self.g_optimizer = optim.Adam(self.G.parameters(), 
                lr=self.eta, betas=(1-self.alpha, 0.999))

        self.d_optimizer = optim.Adam(self.D.parameters(),
            lr=self.eta, betas=(0.5, 0.999))

    def loss(self, x_unlab = None, x_lab = None, y_lab = None):
        """
        Computes the losses for the biven batch of data samples.

        Args:
            x_real: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
            y_real: 'torch.FloatTensor' of shape '(batch_size, K+1)'; data
                labels
            x_unlabeled: torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                unlabeled samples
        
        Returns:
            D_loss: float, discriminator loss
            G_loss: float, generator loss
        """
        batch_size = x_unlab.size()[0]
        x_fake = self.sample(batch_size).cuda()
        

        logSoftMax = nn.LogSoftmax()

        # Losses for generated samples are -log P(y=K+1|x)
        fake_losses = -1*logSoftMax(self.D(x_fake))[:,self.D.K]
        
        if self.mode!="fully":# Losses for unlabeled samples are -log(1-P(y=K+1|x))
            unl_losses =  -1*logOneMinusSoftmax(self.D(x_unlab))[:,self.D.K]
        else: unl_losses = 0

        if self.mode!="uns": # Losses for labeled samples are -log(P(y=yi|x))
            lab_losses = -1*logSoftMax(self.D(x_lab))[:,y_lab]
        else: lab_losses = 0

        #discriminator loss
        d_loss = torch.sum(fake_losses + unl_losses + lab_losses)/batch_size
        #generator loss (original GAN loss, NOT non-saturating)
        g_loss = -1*torch.sum(fake_losses)/batch_size

        if not self.MAP:
            observed_gen = 20000 #batch_size # 50 # What is this hyperparameter doing??
            # It seems like it should be fixed to batch_size
            noise_std = np.sqrt(2 * self.alpha * self.eta)
            g_loss += (self.noiseLoss(self.G, noise_std) / 
                    (observed_gen * self.eta))

        return d_loss, g_loss
        
    @staticmethod
    def noiseLoss(model, std):
        """
        Multiplies all the variables by a normal rv for SGHMC.

        Args:
            std: float, standard deviation of the noise
        
        Returns:
            loss: float, sum of all parameters of the model multiplied by noise
        """
        loss = 0
        std = torch.from_numpy(np.array([std])).float().cuda()
        std = std[0]
        for param in model.parameters():
            means = torch.zeros(param.size()).cuda()
            n = Variable(torch.normal(means, std=std).cuda())
            loss += torch.sum(n * param)
        return loss
    
    def sample(self, n_samples=1):
        """
        Samples from BGAN generator.

        Args:
            n_samples: int, number of samples to produce
        Returns:
            `torch` variable containing `torch.FloatTensor` of shape 
                `(n_samles, z_dim)`
        """
        z = torch.randn(n_samples, self.G.z_dim)
        zv = Variable(z.cuda())
        return self.G(zv)
        
    def step(self, x_real = None, y_real = None, x_unlab = None):
        """
        Makes an SGHMC step for the parameters of BGAN. 

        Args:
            x_batch: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
        """
        self.D.zero_grad()
        self.G.zero_grad()
        d_loss, g_loss = self.loss(x_real,y_real,x_unlab)
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        
        g_loss.backward()
        self.g_optimizer.step()        
        
    def train(self, epochs=100):
        if self.mode == "uns":
            self.unsupervisedTrain(epochs)
        else: raise Exception('Semisupervised and supervised modes not yet written')
    
    def semiTrain(self, epochs):
        print("semisupervised?")
        return
        # labeled_iterator = iter(self.lab_trainloader)
        # dev_iterator = iter(self.devloader)
        # for epoch in range(epochs):
        #     for i, unlabeled_data in enumerate(self.unl_trainloader):

        #         x_unlabeled = Variable(unlabeled_data[0].cuda())
        #         x_lab, y_lab = labeled_iterator.next()
        #         x_lab, y_lab = Variable(x_lab.cuda()), Variable(y_lab.cuda())

        #         self.step(x_unlabeled,x_lab,ylab)

    def unsupervisedTrain(self, epochs):
        losses = np.empty((0,2)) # losses = [Dloss,Gloss]
        for epoch in range(epochs):
            for i, unlabeled_data in enumerate(self.unl_train):
                x_unlabeled = Variable(unlabeled_data[0].cuda())
                self.step(x_unlabeled)

                if i%101==0:
                    getMean = lambda ell: ell.cpu().data.numpy().mean(axis=0)
                    batchLosses = np.array(list(map(getMean, self.loss(x_unlabeled)))).reshape(1,2)
                    losses = np.concatenate((losses,batchLosses))

                if i%500==0:
                    d_loss,g_loss = losses[-1]*.5 + losses[-2]*.25 + losses[-3]*.25 
                    print("[%2d/%d][%3d/%d] Loss_D: %.4f \
                    Loss_G: %.4f" %(epoch,epochs,i,len(self.unl_train),d_loss,g_loss))

            if epoch%5==0:
                tutils.save_image(self.sample(8).data, '%s/fake_samples_epoch_%03d.png'\
                            %(self.save_dir,epoch),normalize=True)

                
                