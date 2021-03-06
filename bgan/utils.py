import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.distributions import Distribution
import numbers

def to_var_gpu(x, volatile=False):
    """ recursively map the elements to variables on gpu """
    if x is None: 
        return x
    if type(x) in [list, tuple]:
        curry = lambda x: to_var_gpu(x, volatile)
        return type(x)(map(curry, x))
    else: 
        return Variable(x, volatile=volatile).cuda()

def to_lambda(x):
    """ Turns constants into constant functions """
    if isinstance(x, numbers.Number):
        return lambda e: x
    else:
        return x

def prettyPrintLog(logDict, *epochIts):
    formatStr = "[%3d/%d][%6d/%d] "
    valuesTuple = epochIts
    for key, val in logDict.items():
        formatStr += (key+": %.3f ")
        valuesTuple += (val,)
    print(formatStr % valuesTuple)

def logOneMinusSoftmax(x):
    """ numerically more stable version of log(1-softmax(x)) """
    max_vals, _ = torch.max(x, 1)
    shifted_x = x - max_vals.unsqueeze(1).expand_as(x)
    exp_x = torch.exp(shifted_x)
    sum_exp_x = exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
    k = x.size()[1]
    batch_size = x.size()[0]
    sum_except_matrix = Variable((torch.ones(k,k) - torch.eye(k)).cuda())
    resized_sum_except_m = sum_except_matrix.squeeze(0).expand(batch_size,k,k)
    sum_except_exp_x = torch.bmm(resized_sum_except_m, exp_x.unsqueeze(2)).squeeze()
    return torch.log(sum_except_exp_x) - torch.log(sum_exp_x)

def wassersteinLoss(lab_logits, y_lab):
    one_hots = torch.zeros(*lab_logits.size()).scatter_(1,y_lab, 1.)
    not_hots = 1 - one_hots
    good = one_hots*lab_logits / torch.sum(one_hots,dim=0,keepdim=True)
    bad = not_hots*lab_logits / torch.sum(not_hots,dim=0,keepdim=True)
    loss_k = torch.sum(good - bad,0)
    lab_loss = torch.mean(loss_k)
    return lab_loss

def wassersteinLoss2(logits, labels):
    #print(labels.unsqueeze(1).size())
    #print(torch.zeros(*logits.size()).size())
    one_hots = torch.zeros(*logits.size()).cuda().scatter_(1,labels.unsqueeze(1),1)
    one_hots = Variable(one_hots.type(torch.cuda.ByteTensor))
    one_hots = one_hots.type(torch.cuda.ByteTensor)
    #print(labels[:5])
    #print(one_hots[:5])
    correctLogits = torch.mean(torch.masked_select(logits, one_hots))
    not_hots = Variable((1 - one_hots.data).type(torch.cuda.ByteTensor))
    incorrectLogits = torch.mean(torch.masked_select(logits, not_hots))
    return incorrectLogits - correctLogits

def grad_penalty(x_real,x_fake):
    beta = torch.rand((x_real.size()[0],1,1,1)).cuda()
    interpolates = beta * x_real + (1-beta)*x_fake
    interpolates = Variable(interpolates, requires_grad = True).cuda()

    disc_interpolates = self.discriminator(interpolates)
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()
    return gradient_penalty

def grad_penalty2(x, y):
    beta = torch.rand((x_real.size()[0],1,1,1)).cuda()
    interpolates = beta * x_real + (1-beta)*x_fake
    interpolates = Variable(interpolates, requires_grad = True).cuda()

    disc_interpolates = self.discriminator(interpolates)
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()
    return gradient_penalty
