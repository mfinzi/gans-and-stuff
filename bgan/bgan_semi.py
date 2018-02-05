import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as tutils


class BganS:
    
    def __init__(self, G, D, dataloaders, save_dir, MAP = False, opt_params=(2e-4,0.01)):
        """ Creates a Bayesian GAN for generator G, discriminator D, and optimizer O.
        """
        self.G = G.cuda()
        self.D = D.cuda()
        (self.lab_trainloader,self.unl_trainloader,self.devloader,
                self.testloader) = dataloaders
        self.MAP = MAP
        self._init_optimizers(opt_params)
            
    def loss(self, x_unlab, x_lab = None, y_lab = None):
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
        #x_lab = x_lab.cuda()
        #if x_unlab: x_unlab = x_unlab.cuda()

        # Generated samples are labeled as fake (last label)
        y_fake = torch.ones(batch_size)*self.D.K_classes
        #y_fake = y_fake.cuda()

        # Binary classification, real labels are 0's, fake are 1
        if not y_lab: y_lab = torch.zeros(batch_size)
        y_lab = y_lab.cuda()
        
            
        cross_entropy = nn.CrossEntropyLoss()
        fake_loss = cross_entropy(self.D(x_fake.detach()), y_fake) #detach for speed
        unl_loss = -1*cross_entropy(self.D(x_unlab),y_fake)
        
        if y_lab: lab_loss = cross_entropy(self.D(x_lab), y_lab)
        if x_unlab: # Not strictly Kosher, need to test
            
        else: 
            unl_loss = 0

        #discriminator loss
        d_loss = fake_loss + unl_loss + lab_loss 
        
        #generator loss
        g_loss = -1*fake_loss

        if not self.MAP:
            noise_std = np.sqrt(2 * self.O.alpha * self.O.eta)
            g_loss += (self.noise_loss(self.generator, noise_std) / 
                    (self.O.gn * self.O.eta))

        return d_loss, g_loss
        
    @staticmethod
    def noise_loss(model, std):
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
    
    def _init_optimizers(opt_params):
        """
        Initializes the optimizers for BGAN.
        """
        eta, alpha = opt_params
        if self.MAP:
            self.g_optimizer = optim.Adam(self.G.parameters(), 
                lr=eta, betas=(0.99, 0.999))
        else self.g_optimizer = optim.Adam(self.G.parameters(), 
                lr=eta, betas=(1-alpha, 0.999))

        self.d_optimizer = optim.Adam(self.D.parameters(),
            lr=eta, betas=(0.99, 0.999))
        
        
        
    def sample(self, n_samples=1):
        """
        Samples from BGAN generator.

        Args:
            n_samples: int, number of samples to produce
        Returns:
            `torch` variable containing `torch.FloatTensor` of shape 
                `(n_samles, z_dim)`
        """
        z = torch.randn(n_samples, self.z_dim)
        zv = Variable(z.cuda())
        return self.generator(zv)
        
    
    def step(self, x_real, y_real = None, x_unlab = None):
        """
        Makes an SGHMC step for the parameters of BGAN. 

        Args:
            x_batch: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
        """

        batchv = Variable(x_batch)
        self.discriminator.zero_grad()
        self.generator.zero_grad()
        d_loss, g_loss = self.loss(x_real,y_real,x_unlab)
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        
#        self.generator.zero_grad()
#        self.discriminator.zero_grad()
#        
#        d_loss, g_loss = self.loss(batchv)
        g_loss.backward()
        self.g_optimizer.step()        
        
    def train(self, epochs=100):
        if self.lab_trainloader:
            self.semi_train(epochs)
        else:
            self.unsupervised_train(epochs)
    
    def semi_train(self, epochs):
        return
        # labeled_iterator = iter(self.lab_trainloader)
        # dev_iterator = iter(self.devloader)
        # for epoch in range(epochs):
        #     for i, unlabeled_data in enumerate(self.unl_trainloader):

        #         x_unlabeled = Variable(unlabeled_data[0].cuda())
        #         x_lab, y_lab = labeled_iterator.next()
        #         x_lab, y_lab = Variable(x_lab.cuda()), Variable(y_lab.cuda())

        #         self.step(x_unlabeled,x_lab,ylab)

    def unsupervised_train(self, epochs):
        losses = np.empty((0,2)) # losses = [Dloss,Gloss]
        for epoch in range(epochs):
            for i, unlabeled_data in enumerate(self.unl_trainloader):
                x_unlabeled = Variable(unlabeled_data[0].cuda())
                self.step(x_unlabeled)

                if i%177==0:
                    batchLosses = self.loss(x_unlabeled).cpu().data.numpy().mean(axis=0)
                    np.concatenate((losses,batchLosses))
                    d_loss,g_loss = losses[-1]
                    print('[%d/%d][%d/%d] Loss_D: %.4f \
                    Loss_G: %.4f %(epoch,epochs,i,len(self.unl_trainloader),d_loss,g_loss))

            if epoch%5==0:
                self.sample(10)
                tutils.save_image(self.sample(10).data, '%s/fake_samples_epoch_%03d.png'\
                            %(save_dir,epoch),normalize=True)

                
                