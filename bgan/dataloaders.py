import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class ShuffleCycleSubsetSampler(Sampler):
    """A cycle version of SubsetRandomSampler with
        reordering on restart """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return self._gen()

    def _gen(self):
        i = len(self.indices)
        while True:
            if i >= len(self.indices):
                perm = np.random.permutation(self.indices)
                i=0
            yield perm[i]
            i+=1
    
    def __len__(self):
        return len(self.indices)

class EmptyLoader(object):
    """A dataloader that loads None tuples, with zero length for convenience"""
    def __next__(self):
        return (None,None)
    def __len__(self):
        return 0
    def __iter__(self):
        return self
        
def getUandLloaders(trainset, amntLabeled, lab_BS, ul_BS, **kwargs):
    """ Returns two cycling dataloaders where the first one only operates on a subset
        of the dataset. AmntLabeled can either be a fraction or an integer """
    numLabeled = amntLabeled
    if amntLabeled <= 1: 
        numLabeled *= len(trainset)
    
    indices = np.random.permutation(len(trainset))
    labIndices = indices[:numLabeled]

    labSampler = ShuffleCycleSubsetSampler(labIndices)
    labLoader = DataLoader(trainset,sampler=labSampler,batch_size=lab_BS,**kwargs)
    if amntLabeled == 0: labLoader = EmptyLoader()

    # Includes the labeled samples in the unlabeled data
    unlabSampler = ShuffleCycleSubsetSampler(indices)
    unlabLoader = DataLoader(trainset,sampler=unlabSampler,batch_size=ul_BS,**kwargs)
        
    return unlabLoader, labLoader

def classBalancedSampleIndices(y, numLabeled):
    uniqueVals = np.unique(y)
    numLabeled = np.floor(numLabeled / len(uniqueVals))*len(uniqueVals)
    classIndices = np.array([np.where(y==val) for val in uniqueVals])
    sampledIndices = np.empty(numLabeled, dtype=np.int64)

    sampledIndices = np.random.choice(classIndices)
    #TODO: Finish this