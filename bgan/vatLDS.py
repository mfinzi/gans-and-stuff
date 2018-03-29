import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def kl_div_withlogits(p_logits, q_logits):
    kl_div = nn.KLDivLoss(size_average=True)
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return kl_div(LSM(q_logits), SM(p_logits))

# def mse(p_logits, q_logits):
#     assert q_logits.size() == p_logits.size()
#     input_softmax = F.softmax(q_logits, dim=1)
#     target_softmax = F.softmax(p_logits, dim=1)
#     b_size = q_logits.size()[0]
#     return F.mse_loss(input_softmax, target_softmax, size_average=False) / b_size

def cross_ent_withlogits(p_logits,q_logits):
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return -1*(SM(p_logits)*LSM(q_logits)).sum(dim=1).mean(dim=0)

def _l2_normalize(d):
    d /= (1e-25 + torch.max(torch.abs(d)))
    norm = torch.sqrt((d**2).sum(3).sum(2).sum(1) + 1e-12)
    d /= norm.view(-1,1,1,1)
    return d

def getNorm(d):
    return torch.mean(torch.sqrt((d**2).sum(3).sum(2).sum(1)))

def getAdvPert(model, X, logits=None, powerIts=1, xi=1e-6):
    # wasTrain = model.training
    # model.eval()
    if logits is None: logits = model(X) # To avoid one additional forward
    #logits = model(X)
    # prepare random unit tensor
    ddata = torch.randn(X.size()).cuda()
    # calc adversarial direction
    for _ in range(powerIts):
        d = Variable(xi*_l2_normalize(ddata), requires_grad=True)
        logit_p = logits.detach()
        perturbed_logits = model(X + d)
        adv_distance = kl_div_withlogits(logit_p, perturbed_logits)
        adv_distance.backward()
        ddata = d.grad.data
        model.zero_grad()

    # if wasTrain: model.train()
    return Variable(_l2_normalize(ddata), requires_grad=False)

def LDSloss(model, X, eps, entMin=False, **kwargs):
    """ LDS loss function
    """
    # wasTrain = model.training
    # model.eval()
    logits = model(X).detach()
    # calc LDS
    r_adv = eps * getAdvPert(model, X, logits, **kwargs)
    perturbed_logits = model(X + r_adv)

    if entMin:
        assert False, "error"
        LDS = cross_ent_withlogits(logits, perturbed_logits)
    else:
        LDS = kl_div_withlogits(logits, perturbed_logits)
    # if wasTrain: model.train()
    return LDS