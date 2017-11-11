import numpy as np
import torch.utils.data as data
import torch
from abc import ABCMeta, abstractmethod
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


class DatasetFromTensors(data.Dataset):

    def __init__(self, X, y=None, labeled=False):
        """
        A dataset from given `X` and `y` torch.Tensors
        Args:
            X: `torch.Tensor`, features of the data samples
            y: `torch.Tensor`, labels of the data samples; if None, and the 
                `labeled` parameter is set to `True`, an error will be rased
            labeled: if `True`, `__getitem__` will return a tuple of `x` and
                `y`; otherwise only `x` is returned
        """
        self.X = X
        self.y = y
        self.N = self.X.shape[0]
        self.labeled = labeled

        if self.y is None and self.labeled:
            raise ValueError("`labeled` set to `True`, but `y` is None")

    def __getitem__(self, index):
        x = self.X[index]
        if self.labeled:
            y = self.y[index]
            return x, y
        else:
            return x

    def __len__(self):
        return self.N


class SynthDataset(DatasetFromTensors):
 
    def __init__(self, x_dim=100, true_z_dim=2, N=10000, num_clusters=10, 
            seed=None, labeled=True, spread=5):
        """
        Synthetic dataset.
        
        Args:
            x_dim: dimensionality of outputs
            true_z_dim: true dimensionality of the distribution
            N: number of data points
            num_clusters: number of clusters (modes) of the distribution
            seed: random seed
            labeled: bool, wether or not to produse class labels for the data
        """
        np.random.seed(seed)
        self.x_dim = x_dim
        self.N = N
        self.true_z_dim = true_z_dim
        self.num_clusters = num_clusters
        self.spread = spread
        X, y = self._generate_points()
        DatasetFromTensors.__init__(self, X, y, labeled)

        
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
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y


#        if self.labeled > 0:
#            
#            self.n_labeled = int(y.size*self.labeled)
#            labeled_indices = np.random.choice(y.shape[0], 
#                    size=self.n_labeled)
#            unlabeled_indices = np.setdiff1d(np.arange(y.shape[0]), 
#                    labeled_indices)
#
#            y_labeled = y[labeled_indices]
#            X_labeled = X[labeled_indices]
#            X_labeled = torch.from_numpy(X_labeled).float()
#            y_labeled = torch.from_numpy(y_labeled).float()
#            self.labeled_dataset = DatasetFromTensors(X_labeled, y_labeled)
#            
#            X_unlabeled = X[unlabeled_indices]
#            X_unlabeled = torch.from_numpy(X_unlabeled).float()
#            self.unlabeled_dataset = DatasetFromTensors(X_unlabeled)


#    def __getitem__(self, index):
#        """
#        Args:
#            index (int): Index
#        Returns:
#            The data point corresponding to the given index
#        """
#        x, y = self.X[index], self.semi_y[index]
#        if self.labeled:
#            return x, y
#        else:
#            return x
#
#    def __len__(self):
#        return self.N


def make_semi_dataset(dataset, labeled_fraction):
    """
    Generates a labeled and an unlabeled dataset from a given labeled dataset.

    This method is meant to be used for generating datasets for semi-supervised 
    classification.

    Args:
        dataset: `DatasetFromTensors` with `labeled` value set to `True`
        labeled_fraction: `float` between 0 and 1, the fraction of labeled data
            points

    Returns:
        A tuple (labeled_ds, unlabeled_ds);
        labeled_ds: `DatasetFromTensors`, a labeled dataset containing a subset
            of sampes from the `dataset` 
        unlabeled_ds: `DatasetFromTensors`, an unlabeled dataset containing the 
        rest of the sampes from the `dataset` 
    """
    
    if not dataset.labeled:
        raise ValueError('The `dataset` must be labeled')

    if not (labeled_fraction > 0) or not (labeled_fraction < 1):
        raise ValueError('The `labeled_fraction` must be between 0 and 1')

    X, y = dataset.X.numpy(), dataset.y.numpy()

    n_labeled = int(y.shape[0] * labeled_fraction)
    labeled_indices = np.random.choice(y.shape[0], size=n_labeled)
    unlabeled_indices = np.setdiff1d(np.arange(y.shape[0]), labeled_indices)

    y_labeled = y[labeled_indices]
    X_labeled = X[labeled_indices]
    X_labeled = torch.from_numpy(X_labeled).float()
    y_labeled = torch.from_numpy(y_labeled).float()
    labeled_ds = DatasetFromTensors(X_labeled, y_labeled, labeled=True)
    
    X_unlabeled = X[unlabeled_indices]
    X_unlabeled = torch.from_numpy(X_unlabeled).float()
    unlabeled_ds = DatasetFromTensors(X_unlabeled, labeled=False)
    return labeled_ds, unlabeled_ds
