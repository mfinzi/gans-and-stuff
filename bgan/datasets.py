import numpy as np
import torch.utils.data as data
import torch
from abc import ABCMeta, abstractmethod
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


class SynthDataset(data.Dataset):
    
    def __init__(self, x_dim=100, true_z_dim=2, N=10000, num_clusters=10, 
            seed=None, labeled=0, spread=5):
        """
        Synthetic dataset.
        
        Args:
            x_dim: dimensionality of outputs
            true_z_dim: true dimensionality of the distribution
            N: number of data points
            num_clusters: number of clusters (modes) of the distribution
            seed: random seed
            labeled: float, fraction of the number of data samples for which the
                label is known; if 0, no labels will be returned
        """
        np.random.seed(seed)
        self.x_dim = x_dim
        self.N = N
        self.true_z_dim = true_z_dim
        self.num_clusters = num_clusters
        self.labeled = labeled
        self.spread = spread
        self._generate_points()
        
    def _generate_points(self):
        """
        Generates data points.
        """
        Xs = []
        num_points = int(self.N / self.num_clusters)
        for i in range(self.num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * self.spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * self.spread
            if i == self.num_clusters - 1:
                num_points += self.N % self.num_clusters
            eps = np.random.randn(num_points, self.true_z_dim)
            X = (eps + cluster_mean).dot(A.T)
            Xs.append(X)
        ys = [np.ones((X.shape[0], 1))*i for (i, X) in enumerate(Xs)]
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        X = (X - X.mean(0)) / (X.std(0))
        self.X = torch.from_numpy(X).float()
        self.true_y = torch.from_numpy(y).float()
        if self.labeled > 0:
            
            self.n_labeled = int(y.size*self.labeled)
            labeled_indices = np.random.choice(y.shape[0], 
                    size=self.n_labeled)
            unlabeled_indices = np.setdiff1d(np.arange(y.shape[0]), 
                    labeled_indices)

            y_labeled = y[labeled_indices]
            X_labeled = X[labeled_indices]
            X_labeled = torch.from_numpy(X_labeled).float()
            y_labeled = torch.from_numpy(y_labeled).float()
            self.labeled_dataset = DatasetFromTensors(X_labeled, y_labeled)
            
            X_unlabeled = X[unlabeled_indices]
            X_unlabeled = torch.from_numpy(X_unlabeled).float()
            self.unlabeled_dataset = DatasetFromTensors(X_unlabeled)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            The data point corresponding to the given index
        """
        x, y = self.X[index], self.semi_y[index]
        if self.labeled:
            return x, y
        else:
            return x

    def __len__(self):
        return self.N

class DatasetFromTensors(data.Dataset):

    def __init__(self, X, y=None):
        """
        A dataset from given `X` and `y` torch.Tensors
        """
        self.X = X
        self.y = y
        self.N = self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        if self.y is not None:
            y = self.y[index]
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
