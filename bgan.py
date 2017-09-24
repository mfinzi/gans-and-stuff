import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
class BGAN:
    
    def __init__(self, generator, discriminator, 
                 generator_prior, discriminator_prior, num_data,
                 J=1, M=1, eta=2e-4, alpha=0.01, observed_gen=50):
        """
		Creates a Bayesian GAN for the given generator and discriminator.
		
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
				hyper-parameter affects the uncertainty in the generator weights
        """
        super(BGAN, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.generator_prior = generator_prior
        self.discriminator_prior = discriminator_prior
        
        self.x_dim = generator.output_dim
        self.z_dim = generator.input_dim

        self.num_gen = J
        self.num_mcmc = M
        self.eta = eta
        self.alpha = alpha
        self.num_data = num_data
        self.observed_gen = observed_gen
            
        self.K = discriminator.K
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
        
        d_logits_real = self.discriminator(x_real)[:, 0]
        d_logits_fake = self.discriminator(x_fake)[:, 0]
        
        y_real = Variable(torch.ones(batch_size))
        y_fake = Variable(torch.zeros(batch_size))

        bce = nn.BCELoss()
        bce_real = bce(d_logits_real, y_real)

        bce_fake = bce(d_logits_fake, y_fake)
        
        noise_std = np.sqrt(2 * self.alpha * self.eta)
        
        #discriminator loss
        d_loss = -(bce_real + bce_fake) * self.eta 
#         d_loss += (self.discriminator_prior.log_density(self.discriminator) 
#						* self.eta #/ self.num_data)
#         d_loss += self.noise(self.generator, noise_std) #/ self.num_data
        d_loss *= -1.
        
        #generator loss
        g_loss = bce_fake * self.eta
        g_loss += (self.generator_prior.log_density(self.generator) 
					* self.eta / self.obsered_gen)
        g_loss += self.noise(self.generator, noise_std) / self.observed_gen
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
        for param in model.parameters():
            n = Variable(torch.normal(0, std=std*torch.ones(param.size())))
            loss += torch.sum(n * param)
        return loss
    
    def _init_optimizers(self):
		"""
        Initializes the optimizers for BGAN.
        """
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=1, 
							momentum=(1 - self.alpha))
        self.g_optimizer = optim.SGD(self.generator.parameters(), lr=1, 
							momentum=(1 - self.alpha))
        
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
        return self.generator(zv)
        
    
    def step(self, x_batch):
		"""
        Makes an SGHMC step for the parameters of BGAN. 

        Args:
            x_batch: `torch.FloatTensor` of shape `(batch_size, x_dim)`; data
                samples
        """

        batchv = Variable(batch)
        self.discriminator.zero_grad()
        self.generator.zero_grad()
        d_loss, g_loss = self.loss(batchv)
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        
        g_loss.backward()
        self.g_optimizer.step()        
        
