import numpy as np
from abc import ABCMeta, abstractmethod
import torch.utils.data as data

class SynthDataset(data.Dataset):
    
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
        self._generate_points()
        
    def _generate_points(self):
        """
        Generates data points.
        """
        Xs = []
        num_points = int(self.N / self.num_clusters)
        for i in range(self.num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            if i == self.num_clusters - 1:
                num_points += self.N % self.num_clusters
            eps = np.random.randn(num_points, self.true_z_dim)
            X = (eps + cluster_mean).dot(A.T)
            Xs.append(X)
        X_raw = np.concatenate(Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            The data point corresponding to the given index
        """
        x = self.X[index]
        return x

    def __len__(self):
        return self.N
        
