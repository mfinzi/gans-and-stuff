import torch
from abc import ABCMeta, abstractmethod

class Prior:
    """
    Abstract class representing prior distribution over NN weights.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def log_density(self, model):
        """
        Computes prior log-density of the `model`.
		Args:
			model: torch.nn.Module instance
        """
        raise NotImplementedError


class FactorizedNormalPrior(Prior):
    
    def __init__(self, std=1.0):
        """
		Factorized Normal prior over the parameters of the model.
		
	    Args:
            std: standard diviation of the normal prior
		"""
        self.std = std
    
    def log_density(self, model):
        log_prior = 0.
        for param in model.parameters():
            log_prior += -torch.sum(param**2 / (2 * self.std**2))
        return log_prior
