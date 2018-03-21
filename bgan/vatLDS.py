import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kl_div_withlogits(p_logits,q_logits):
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return (SM(p_logits)*(LSM(p_logits) - LSM(q_logits))).sum(dim=1).mean(dim=0)

def kl_div2(p_logits, q_logits):
    kl_div = nn.KLDivLoss(size_average=False)
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return kl_div(LSM(q_logits), SM(p_logits))/q_logits.size()[0]

def mse(p_logits, q_logits):
    assert q_logits.size() == p_logits.size()
    input_softmax = F.softmax(q_logits, dim=1)
    target_softmax = F.softmax(p_logits, dim=1)
    b_size = q_logits.size()[0]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / b_size

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
    wasTrain = model.training
    model.eval()
    #if logits is None: logits = model(X) # To avoid one additional forward
    #logits = model(X)
    # prepare random unit tensor
    d = torch.randn(X.size())
    d = Variable(_l2_normalize(d).cuda(), requires_grad=True)
    # calc adversarial direction
    #for _ in range(powerIts):
    #perturbed_logits = model(X + xi*d)
    #adv_distance = kl_div_withlogits(logits.detach(), perturbed_logits)
    #adv_distance = kl_div2(logits.detach(), perturbed_logits)
    #adv_distance = mse(logits.detach(), perturbed_logits)
    #adv_distance.backward()
    #print(getNorm(d.grad))
    #d = Variable(_l2_normalize(d.grad.data).cuda(), requires_grad=False)
    #model.zero_grad()
    d.requires_grad = False
    #print(getNorm(d))
    if wasTrain: model.train()
    return d

def LDSloss(model, X, eps, entMin=False, **kwargs):
    """ LDS loss function
    """
    wasTrain = model.training
    model.eval()
    #logits = model(X)
    # calc LDS
    r_adv = eps * getAdvPert(model, X, None, **kwargs)
    perturbed_logits = model(X + r_adv)
    logits = model(X)
    if entMin:
        LDS = cross_ent_withlogits(logits.detach(), perturbed_logits)
    else:
        #LDS = mse(logits.detach(), perturbed_logits)
        LDS = kl_div_withlogits(logits.detach(), perturbed_logits)
        #LDS = kl_div2(logits.detach(), perturbed_logits)
    if wasTrain: model.train()
    return LDS