import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn as cudnn

class FixedSizeDataset(torch.utils.data.Dataset):
    """
    Dataset with a fixed size, that allows appending new data.

    This class is meant to be used with BGANNG.
    """
    
    def __init__(self, maxlen):
        """
        Creates a FixedSozeDataset object.

        Args:
            maxlen: maximum size of the dataset
        """
        self.elems = []
        self.maxlen = maxlen
        
    def __getitem__(self, index):
        return self.elems[index]
    
    def append(self, elem):
        self.elems.append(elem)
        if len(self) > self.maxlen:
            self.elems = self.elems[1:]
    
    def __len__(self):
        return len(self.elems)


class BGANNG:
    
    def __init__(self, generator, generator_prior, discriminator, num_data, 
                eta=2e-4, alpha=0.01, max_fake=10000, disc_lr=1e-3, 
                observed_gen=50, cuda=False, MAP=False):
        """
        Creates a Bayesian GAN with no generator.
        
        Args:
            generator: `torch.nn.Module` instance, generator network
            discriminator: `torch.nn.Module` instance, discriminator network
            generator_prior: `Prior` instance, prior over the weights of
                generator network
            discriminator_prior: `Prior` instance, prior over the weights of
                discriminator network
            J: int, number number of models to average
            M: int, number of MCMC steps per iteration
            eta: float, learning rate; also affects noise variance in SGHMC
            alpha: float, momentum variable; also affects noise variance in 
                SGHMC
            observed_gen: number of data observed by the generator; this 
                hyper-parameter affects the uncertainty in the generator
        """

        self.discriminator = discriminator
        self.generator = generator

        self.cuda = cuda

        if self.cuda:
            print('Moving generator and discriminator to GPU')
            self.discriminator.cuda()
            self.generator.cuda() # should I do this

        self.MAP = MAP
        
        self.eta = eta
        self.alpha = alpha
        self.num_data = num_data
        self.disc_lr = disc_lr
        self.generator_prior = generator_prior
            
        self._init_optimizers()
        self.fake_dataset = FixedSizeDataset(max_fake)
        self.fake_dataset.append(np.copy(self.generator.forward().cpu().data.numpy())[0, :])
        self.fake_batch_loader = torch.utils.data.DataLoader(self.fake_dataset, 
                                                            batch_size=64, shuffle=True)
        self.fake_batch_generator = self.get_fake_batch()
        self.gen_observed = observed_gen
        
    def get_fake_batch(self):
        while True:
            for batch in self.fake_batch_loader:
                yield Variable(batch.float())
            
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
        fake_batch = next(self.fake_batch_generator)
        x_gen = self.generator.forward()
        x_real = x_batch
        
        if self.cuda:
            fake_batch = fake_batch.cuda()
            x_real = x_real.cuda()
            x_gen = x_gen.cuda()
        
        d_logits_real = self.discriminator(x_real)
        d_logits_fake = self.discriminator(fake_batch)
        d_logits_gen = self.discriminator(x_gen)
        y_real = Variable(torch.ones(batch_size))
        y_fake = Variable(torch.zeros(fake_batch.size()[0]))
        y_gen = Variable(torch.zeros(x_gen.size()[0]))
        
        if self.cuda:
            y_real = y_real.cuda()
            y_fake = y_fake.cuda()
            y_gen = y_gen.cuda()
        
        bce = nn.BCELoss()
        if self.cuda:
            bce.cuda()

        bce_real = bce(d_logits_real, y_real)
        bce_fake = bce(d_logits_fake, y_fake)
        bce_gen = bce(d_logits_gen, y_gen)
        
        d_loss = -(bce_real + bce_fake) * self.disc_lr
        d_loss *= -1.
        
        #generator loss
        noise_std = np.sqrt(2 * self.alpha / self.eta)
        g_loss = -bce(d_logits_gen, y_real[0])
#        print(y_real[0])
#        print(d_logits_gen)
#        print(g_loss)
#        g_loss = torch.mean(torch.log(d_logits_gen[0])) * self.eta
#        g_loss -= torch.mean(torch.log(1 - d_logits_gen[0])) * self.eta
        g_noise = self.noise(self.generator, noise_std) / self.gen_observed
        g_prior = (self.generator_prior.log_density(self.generator) / 
                self.gen_observed)
        if self.cuda:
            g_prior = g_prior.cuda()
            g_noise = g_noise.cuda()

#        g_loss += g_prior
#        g_loss += g_noise
        g_loss *= -1.
#        self.d_loss_fake = (bce_fake).data.cpu().numpy()[0]
#        self.d_loss_real = bce_real.data.cpu().numpy()[0]
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
#            n = Variable(torch.normal(0, std=std*torch.ones(param.size())).cuda())
            n = Variable(torch.normal(means, std=std).cuda())
            loss += torch.sum(n * param)
        return loss
    
    def _init_optimizers(self):
        """
        Initializes the optimizers for BGAN.
        """
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                lr=self.disc_lr, betas=(0.5, 0.999))
#        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=1)
        if self.MAP:
            self.g_optimizer = optim.Adam(self.generator.parameters(), 
                    lr=self.eta, betas=(0.5, 0.999))
        else:
            self.g_optimizer = optim.Adam(self.generator.parameters(), 
                    lr=self.eta, betas=(1-self.alpha, 0.999))
#        self.g_optimizer = optim.SGD(self.generator.parameters(), lr=1, 
#                momentum=(1 - self.alpha))
        
    
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
        
        g_loss.backward()
        self.g_optimizer.step()     
        