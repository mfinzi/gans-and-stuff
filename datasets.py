import numpy as np
from abc import ABCMeta, abstractmethod

class Dataset:
	"""
	Abstract base class representing a dataset and defyning the interface.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def next_batch(self, batch_size):
		pass

class SynthDataset(Dataset):
    
    def __init__(self, x_dim=100, true_z_dim=2, N=10000, num_clusters=10, seed=None):
        """
        Synthetic dataset.
        
        Args:
            x_dim: dimensionality of outputs
            true_z_dim: true dimensionality of the distribution
            N: number of data points
            num_clusters: number of clusters (modes) of the distribution
            seed: random seed
        """
        np.random.seed(None)
        self.x_dim = x_dim
        self.N = N
        self.true_z_dim = true_z_dim
        self.num_clusters = num_clusters
        self.generate_points()
        
    def generate_points(self):
        """
        Generates data points.
        """
        Xs = []
        for _ in range(self.num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            eps = np.random.randn(int(self.N / self.num_clusters), self.true_z_dim)
            X = (eps + cluster_mean).dot(A.T)
            Xs.append(X)
        X_raw = np.concatenate(Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        
    def next_batch(self, batch_size):
        """
        Returns a batch of datapoints.
        """
        rand_idx = np.random.choice(range(self.N), size=(batch_size,), replace=False)
        return self.X[rand_idx]
