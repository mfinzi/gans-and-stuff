import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class BGAN:
    
    def __init__(self, generator, discriminator, generator_prior, eta=2e-4, 
            alpha=0.01, gen_observed=50, disc_lr=None, MAP=False, cuda=False):
        """
        Creates a Bayesian GAN for the given generator and discriminator.
        
        Args:
            generator: `torch.nn.Module` instance, generator network
            discriminator: `torch.nn.Module` instance, discriminator network
            generator_prior: `Prior` instance, prior over the weights of
                generator network
            eta: float, learning rate; also affects noise variance in SGHMC
            alpha: float, momentum variable; also affects noise variance in 
                SGHMC
            gen_observed: number of data observed by the generator; this 
                hyper-parameter affects the uncertainty in the generator weights
            MAP: bool, if `True` a map estimate is used instead of sampling
            cuda: bool, if `True` CUDA is used to run the computations on GPU
        """
        self.generator = generator
        self.discriminator = discriminator
        self.generator_prior = generator_prior
        self.z_dim = generator.input_dim

        self.cuda = cuda
        if self.cuda:
            print('Moving generator and discriminator to GPU')
            self.discriminator.cuda()
            self.generator.cuda()

        self.eta = eta
        if disc_lr is None:
            self.disc_lr = eta
        else:
            self.disc_lr = disc_lr
        self.alpha = alpha
        self.observed_gen = gen_observed 
        self.MAP = MAP
            
        self._init_optimizers()
            
    def loss(self, x_batch):
        """
        Computes the losses for the biven batch of data samples.

        Args:
            x_batch: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
        
        Returns:
            d_loss: float, discriminator loss
            g_loss: float, generator loss
        """
        batch_size = x_batch.size()[0]
        x_fake = self.sample(batch_size)
        x_real = x_batch
        
        if self.cuda:
            x_fake = x_fake.cuda()
            x_real = x_real.cuda()

        d_logits_real = self.discriminator(x_real)
        d_logits_fake = self.discriminator(x_fake)
        y_real = Variable(torch.ones(batch_size))
        y_fake = Variable(torch.zeros(batch_size))

        if self.cuda:
            y_real = y_real.cuda()
            y_fake = y_fake.cuda()
        
        bce = nn.BCELoss()
        bce = nn.BCELoss()
        if self.cuda:
            bce.cuda()

        bce_real = bce(d_logits_real, y_real)
        bce_fake = bce(d_logits_fake, y_fake)
        
        #discriminator loss
        d_loss = -(bce_real + bce_fake) 
        d_loss *= -1.
        
        #generator loss
        g_loss = -bce(d_logits_fake, y_real)
        g_loss += (self.generator_prior.log_density(self.generator) / 
                self.observed_gen)
        if not self.MAP:
            noise_std = np.sqrt(2 * self.alpha * self.eta)
            g_loss += (self.noise(self.generator, noise_std) / 
                    (self.observed_gen * self.eta))
        g_loss *= -1.
        return d_loss, g_loss
        
    @staticmethod
    def noise(model, std):
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
    
    def _init_optimizers(self):
        """
        Initializes the optimizers for BGAN.
        """
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                lr=self.disc_lr, betas=(0.5, 0.999))
        if self.MAP:
            self.g_optimizer = optim.Adam(self.generator.parameters(), 
                    lr=self.eta, betas=(0.5, 0.999))
        else:
            self.g_optimizer = optim.Adam(self.generator.parameters(), 
                    lr=self.eta, betas=(1-self.alpha, 0.999))
        
        
    def sample(self, n_samples=1):
        """
        Samples from BGAN generator.

        Args:
            n_samples: int, number of samples to produce
        Returns:
            `torch` variable containing `torch.FloatTensor` of shape 
                `(n_samles, z_dim)`
        """
        z = torch.rand(n_samples, self.z_dim)
        zv = Variable(z)
        if self.cuda:
            zv = zv.cuda()
        return self.generator(zv)
        
    
    def step(self, x_batch):
        """
        Makes an SGHMC step for the parameters of BGAN. 

        Args:
            x_batch: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
        """

        batchv = Variable(x_batch)
        self.discriminator.zero_grad()
        self.generator.zero_grad()
        d_loss, g_loss = self.loss(batchv)
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        
#        self.generator.zero_grad()
#        self.discriminator.zero_grad()
#        
#        d_loss, g_loss = self.loss(batchv)
        g_loss.backward()
        self.g_optimizer.step()        
        
