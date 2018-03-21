import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils
import tensorboardX
import itertools
from bgan.utils import to_var_gpu, prettyPrintLog, logOneMinusSoftmax
from bgan.cnnTrainer import CnnTrainer


class GanTrainer(CnnTrainer):
    
    def __init__(self, G, D, datasets, amntLab, save_dir, batchSize = 64, 
                    lr=1e-4, momentum=.5, featMatch = False, 
                    bayesianG = False, bayesianD = False, genObserved=50000,
                    lipPenalty = False):
        """ Creates a Bayesian GAN for generator G, discriminator D, and optimizer O.
        """
        self.writer = tensorboardX.SummaryWriter(save_dir)

        assert torch.cuda.is_available(), "CUDA or GPU working"
        self.G = G.cuda(); self.writer.add_text('ModelSpec','Generator: '+type(G).__name__)
        self.D = D.cuda(); self.writer.add_text('ModelSpec','Discriminator: '+type(D).__name__)
        assert D.numClasses > 1, "D must have 2 or more output channels, 2 for Binary Class"
        
        self.hypers = {'lr':lr, 'momentum':momentum, 'batchSize':batchSize,
                        'amntLab':amntLab, 'featMatch':featMatch, 
                        'bayesianG':bayesianG, 'bayesianD':bayesianD,
                        'genObserved':genObserved, 'lipPenalty':lipPenalty}

        self.mode = self.getMode() #{"uns","semi","fully"}
        self.d_optimizer, self.g_optimizer = self.initOptimizers()
        self.lab_iter, self.dev_iter, self.test_iter, self.unl_iter \
                    = self.getDataIters(datasets, batchSize, amntLab)
        
    def getMode(self):
        if self.hypers['amntLab']==0: mode = "uns"
        elif self.hypers['amntLab']==1: mode = "fully"
        else: mode = "semi"
        return mode

    def initOptimizers(self):
        d_optim = optim.Adam(self.D.parameters(),
            lr=self.hypers['lr'], betas=(self.hypers['momentum'], 0.999))

        g_optim = optim.Adam(self.G.parameters(), 
            lr=self.hypers['lr'], betas=(self.hypers['momentum'], 0.999))
        return d_optim, g_optim

    def loss(self, x_unlab, x_lab, y_lab):
        """
        """
        # TODO: decouple the batch sizes for possible adjustments
        if self.mode != 'fully': batch_size = x_unlab.size()[0]
        else: batch_size = x_lab.size()[0]
        
        x_fake = self.sample(batch_size)
        fake_logits = self.D(x_fake)
        logSoftMax = nn.LogSoftmax(dim=1)
        # Losses for generated samples are -log P(y=K+1|x)
        fake_losses = -1*logSoftMax(fake_logits)[:,self.D.numClasses-1]

        unl_losses, lab_losses = 0, 0
        if self.mode!="fully":# Losses for unlabeled samples are -log(1-P(y=K+1|x))
            x_real = x_unlab; real_logits = self.D(x_real)
            unl_losses =  -1*logOneMinusSoftmax(real_logits)[:,self.D.numClasses-1]

        if self.mode!="uns": # Losses for labeled samples are -log(P(y=yi|x))
            x_real = x_lab; real_logits = self.D(x_lab)
            lab_losses = nn.CrossEntropyLoss(reduce=False)(real_logits,y_lab)
        
        #discriminator loss
        d_loss = torch.mean(fake_losses + unl_losses + lab_losses)
        if self.hypers['lipPenalty']:
            d_loss += self.lipschitzPenalty(x_unlab, x_fake, real_logits, fake_logits)

        if self.hypers['featMatch']:
            if self.mode=="fully": x_comp = x_lab
            else: x_comp = x_unlab
            real_features = torch.mean(self.D(x_comp, getFeatureVec=True),0)
            fake_features = torch.mean(self.D(x_fake, getFeatureVec=True),0)
            g_loss = 1000*torch.mean(torch.abs(real_features-fake_features))
        else: #generator loss (non-saturating loss) -log(1-P(y=K+1|x))
            g_losses = -1*logOneMinusSoftmax(fake_logits)[:,self.D.numClasses-1]
            g_loss = torch.mean(g_losses)

        if self.hypers['bayesianG']:
            g_loss += self.sghmcNoiseLoss(self.G)
        if self.hypers['bayesianD']:
            d_loss += self.sghmcNoiseLoss(self.D)

        return d_loss, g_loss

    def lipschitzPenalty(self,x_real,x_fake,real_logits,fake_logits):
        imgDists = (x_real - x_fake).view(x_real.size(0),-1).norm(p=2,dim=1)
        logitDists = (real_logits - fake_logits).norm(p=1,dim=1)
        oneSided = F.relu(logitDists/imgDists -1)
        penalty = torch.mean(oneSided*oneSided)
        return penalty
        
    def sghmcNoiseLoss(self, model):
        noise_std = np.sqrt(2 * (1-self.hypers['momentum']) * self.hypers['lr'])
        std = torch.from_numpy(np.array([noise_std])).float().cuda()[0]
        loss = 0
        for param in model.parameters():
            means = torch.zeros(param.size()).cuda()
            n = Variable(torch.normal(means, std=std).cuda())
            loss += torch.sum(n * param)
        corrected_loss = loss / (self.hypers['genObserved'] * self.hypers['lr'])
        return corrected_loss
    
    def getNoise(self, n_samples=1):
        return Variable(torch.randn(n_samples, self.G.z_dim).cuda())

    def sample(self, n_samples=1):
        return self.G(self.getNoise(n_samples))
        
    def step(self, x_unlab, x_lab, y_lab):
        tensors = tuple(map(to_var_gpu,(x_unlab,x_lab,y_lab)))
        self.D.zero_grad()
        self.G.zero_grad()
        d_loss, g_loss = self.loss(*tensors)
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        g_loss.backward()
        self.g_optimizer.step()
        return d_loss, g_loss
        
    def batchPredAcc(self, x_real, y_real):
        x_real, y_real = to_var_gpu(x_real), to_var_gpu(y_real)
        self.D.eval()
        fullLogits = self.D(x_real)
        # Exclude the logits for generated class in predictions
        notFakeLogits = fullLogits[:,:self.D.numClasses-1]
        predictions = notFakeLogits.max(1)[1].type_as(y_real)
        correct = predictions.eq(y_real).cpu().data.numpy().mean()
        self.D.train()
        return correct

    def train(self, numEpochs = 100):
        self.writeHypers()
        print("Starting "+self.mode+"Supervised Training")
        fixedNoise = self.getNoise(32)
        cdev_iter = itertools.cycle(self.dev_iter)

        numBatchesPerEpoch = max(len(self.unl_iter),len(self.lab_iter))
        numSteps = numEpochs*numBatchesPerEpoch

        for step in range(numSteps):
            # Get the input data and run optimizer step
            x_unlab, _ = next(self.unl_iter)
            x_lab, y_lab = next(self.lab_iter)
            dloss, gloss = self.step(x_unlab, x_lab, y_lab)

            if step%100==0:
                # Add logging data to tensorboard writer
                logData = { 'G_loss': gloss.cpu().data[0],
                            'D_loss': dloss.cpu().data[0],}
                if self.mode!="uns":
                    logData['Train_Acc'] = self.batchPredAcc(x_lab, y_lab)
                    logData['Val_Acc'] = self.batchPredAcc(*next(cdev_iter))
                self.writer.add_scalars('metrics', logData, step)
            
            if step%1000==0:
                # Print the logdata and write out generated images (on fixed noise)
                epoch = step / numBatchesPerEpoch
                prettyPrintLog(logData, epoch, numEpochs, step, numSteps)
                fakeImages = self.G(fixedNoise).cpu().data
                self.writer.add_image('fake_samples', 
                        vutils.make_grid(fakeImages, normalize=True), step)
        if self.mode!=uns: print('Devset Acc: %.3f'%self.getDevsetAccuracy())       

