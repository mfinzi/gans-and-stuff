import numpy as np
import torch.utils.data as data
import torch
from abc import ABCMeta, abstractmethod
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class SynthDataset(data.Dataset):
    
    def __init__(self, x_dim=100, true_z_dim=2, N=10000, num_clusters=10, 
            seed=None, labeled=False):
        """
        Synthetic dataset.
        
        Args:
            x_dim: dimensionality of outputs
            true_z_dim: true dimensionality of the distribution
            N: number of data points
            num_clusters: number of clusters (modes) of the distribution
            seed: random seed
            labeled: bool, if True then labels are returned for each sample;
                the number of labels is equal to the number of clusters
        """
        np.random.seed(seed)
        self.x_dim = x_dim
        self.N = N
        self.true_z_dim = true_z_dim
        self.num_clusters = num_clusters
        self._generate_points()
        self.labeled = labeled
        
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
        ys = [np.ones((X.shape[0], 1))*i for (i, X) in enumerate(Xs)]
        X_raw = np.concatenate(Xs)
        y_raw = np.concatenate(ys)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(y_raw).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            The data point corresponding to the given index
        """
        x, y = self.X[index]
        if self.labeled:
            return x, y
        else:
            return x

    def __len__(self):
        return self.N
        

class DigitsDataset(data.Dataset):

    def __init__(self):
        digits = load_digits()
        X = digits.data
        self.scaler_x = StandardScaler()
        X = self.scaler_x.fit_transform(X)

        self.N = X.shape[0]
        self.X = torch.from_numpy(X).float()
        
    def __getitem__(self, index):
        x = self.X[index]
        return x.view(1, 8, 8)

    def __len__(self):
        return self.N

    def inverse_transform(self, x):
        x = x.view(-1, 64).numpy()
        x = self.scaler_x.inverse_transform(x)
        return x.reshape([-1, 1, 8, 8])
