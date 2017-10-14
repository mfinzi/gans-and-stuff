import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

import argparse
import time
import os
import pprint
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from datasets import SynthDataset
from synth_utils import js_div, kl_div, pca
from priors import FactorizedNormalPrior
from bgan_nogen import BGANNG, FixedSizeDataset


parser = argparse.ArgumentParser(description='BGAN-NG synthetic experiments')
parser.add_argument('--x_dim',
                    type=int,
                    default=10,
                    help='dim of x for synthetic data')
parser.add_argument('--train_iter',
                     type=int,
                     default=1000,
                     help='number of training epochs')
parser.add_argument('--gen_observed',
                     type=int,
                     default=None,
                     help='number of data "observed" by generator')
parser.add_argument('--out_dir',
                    default="/tmp/synth_results",
                    help='path of where to store results')
parser.add_argument('--with_gen',
                    type=bool,
                    default=True,
                    help='wether or not to use generator')
parser.add_argument('--z_dim',
                    type=int,
                    default=2,
                    help='dim of z for generator')
parser.add_argument('--N',
                    type=int,
                    default=1000,
                    help='number of data points')


def thinning(sample_arr, freq=100):
    """
    Thinning.
    """
    new_sample_arr = []
    for i, sample in enumerate(sample_arr):
        if not i % freq:
            new_sample_arr.append(sample)
    return np.array(new_sample_arr)


def weights_init(m):
    """
    Weight initializer
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class NG(nn.Module):
    
    def __init__(self, num_samples, shape=None, init=None):
        """
        Generator analogue for BGAN-NG. 
        """
        super(NG, self).__init__()
        self.build_net(shape, num_samples, init)
        self.output_dim = shape
    
    def build_net(self, shape, num_samples, init):
        if init is None:
            z = torch.from_numpy(np.random.normal(size=[num_samples] + shape)).float()
        else:
            z = torch.from_numpy(init).float()
        self.z_v = Variable(z, requires_grad=True)
    
    
    def forward(self):
        output = self.z_v
        return output
    
    def parameters(self):
        return [self.z_v]


class DiscriminatorNetwork(nn.Module):
    def __init__(self, x_dim, K, h_dim):
        """
        Discriminator
        """
        super(DiscriminatorNetwork, self).__init__()
        self.build_net(x_dim, K, h_dim)
        self.x_dim = x_dim
        self.K = K
    
    def build_net(self, x_dim, K, h_dim):
        self.network = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, K),
            nn.Softmax()
        )
    
    def forward(self, input):
        # TODO: understand parallelism?
        output = self.network(input)
        return output


args = parser.parse_args()
x_dim = args.x_dim
disc_h_dim = 500
if args.gen_observed is None:
    gen_observed = args.N / 10
else:
    gen_observed = args.gen_observed

if not os.path.exists(args.out_dir):
    print("Creating %s" % args.out_dir)
    os.makedirs(args.out_dir)
results_path = os.path.join(args.out_dir, "experiment_%i" % (int(time.time())))
os.makedirs(results_path)
with open(os.path.join(results_path, "args.txt"), "w") as hf:
    hf.write("Experiment settings:\n")
    hf.write("%s\n" % (pprint.pformat(args.__dict__)))

data = SynthDataset(x_dim=x_dim, N=args.N, num_clusters=3, seed=0)
dataloader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)

disc = DiscriminatorNetwork(x_dim, 2, disc_h_dim)
if args.with_gen:
    prior = FactorizedNormalPrior(std=5.)
    init = data[0].numpy()[None, :]
    gen = NG(1, [x_dim], init=init)
    gen.apply(weights_init)
    disc.apply(weights_init)
    bgan = BGANNG(gen, prior, disc, alpha=.01, eta=1e-4, num_data=len(data), 
            disc_lr=1e-2, observed_gen=gen_observed)
else:
    raise NotImplementedError

samples = []
for epoch in range(args.train_iter):
    for i, data_ in enumerate(dataloader, 0):
        batch = data_.float()
        bgan.step(batch)
        samples.append(np.copy(gen.z_v.data.numpy()))
        bgan.fake_dataset.append(np.copy(gen.z_v.data.numpy()[0, :]))
    if not epoch%100:
        print(epoch)
        print(bgan.d_loss_fake)
        print(bgan.d_loss_real)
    if not (epoch+1)%500:
        samples_arr = np.vstack(samples)
        x_r, x_f = pca(data.X.numpy(), samples_arr)
        plt.figure(figsize=(15, 10))
        plt.plot(x_r[:, 0], x_r[:, 1], 'bo')
        plt.plot(x_f[:, 0], x_f[:, 1], '-r')
        plt.plot(x_f[-1, 0], x_f[-1, 1], 'ro', markersize=10)
        plt.savefig(results_path+"/pca_trajectory_%i.png" % (epoch+1))

samples_arr = np.vstack(samples)
thinned_samples = thinning(x_f, freq=100)
plt.figure(figsize=(15, 10))
plt.plot(x_r[:, 0], x_r[:, 1], 'bo')
plt.plot(thinned_samples[:, 0], thinned_samples[:, 1], 'ro')
plt.savefig(results_path+"/pca_thinned.png")

plt.figure(figsize=(15, 10))
plt.plot(x_r[:, 0], x_r[:, 1], 'bo')
plt.plot(thinned_samples[-500:, 0], thinned_samples[-500:, 1], 'ro')
plt.savefig(results_path+"/pca_thinned_last500.png")

plt.figure(figsize=(15, 10))
plt.plot(x_r[:, 0], x_r[:, 1], 'bo')
plt.plot(thinned_samples[-100:, 0], thinned_samples[-100:, 1], 'ro')
plt.savefig(results_path+"/pca_thinned_last100.png")
