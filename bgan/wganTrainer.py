import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from bgan.ganTrainer import GanTrainer

class WganTrainer(GanTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def loss(self, x_unlab = None, x_lab = None, y_lab = None):
        """
        """
        # TODO: decouple the batch sizes for possible adjustments
        if self.mode != 'fully': batch_size = x_unlab.size()[0]
        else: batch_size = x_lab.size()[0]
        
        x_fake = self.sample(batch_size)
        fake_logits = self.D(x_fake)
        unl_logits = self.D(x_unlab)
        lab_logits = self.D(x_lab)
        # Losses for generated samples are -log P(y=K+1|x)
        fake_loss = torch.mean(fake_logits[:,self.D.numClasses-1])

        

        unl_losses, lab_losses = 0, 0
        if self.mode!="uns": # Losses for labeled samples are -log(P(y=yi|x))
            lab_losses = self.labeledLoss(self, lab_logits, y_lab)
        
        if self.mode!="fully":# Losses for unlabeled samples are -log(1-P(y=K+1|x))
            unl_losses = 0

        
        
        #discriminator loss
        d_loss = torch.mean(fake_losses + unl_losses + lab_losses)

        if self.hypers['featMatch']:
            if self.mode=="fully": x_comp = x_lab
            else: x_comp = x_unlab
            real_features = torch.mean(self.D(x_comp, getFeatureVec=True),0)
            fake_features = torch.mean(self.D(x_fake, getFeatureVec=True),0)
            g_loss = 1000*torch.mean(torch.abs(real_features-fake_features))
        else: #generator loss (non-saturating loss) -log(1-P(y=K+1|x))
            g_losses = -1*logOneMinusSoftmax(fake_logits)[:,self.D.numClasses-1]
            g_loss = torch.mean(g_losses)

        if self.hypers['bayesian']:
            noise_std = np.sqrt(2 * (1-self.hypers['momentum']) * self.hypers['lr'])
            g_loss += (self.noiseLoss(self.G, noise_std) / 
                    (self.hypers['genObserved'] * self.hypers['lr']))

        return d_loss, g_loss

    def grad_penalty(self,x_real,x_fake):
        # Same as in wgan gp paper
        # randomly sample on lines connecting real and fake samples
        beta = torch.rand(x_real.size())
        if self.cuda: beta = beta.cuda()
        interpolates = beta * x_real + (1-beta)*x_fake
        if self.cuda: interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad = True)

        disc_interpolates = self.discriminator(interpolates)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).cuda() if self.cuda else torch.ones(
                                disc_interpolates.size()),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()
        return gradient_penalty