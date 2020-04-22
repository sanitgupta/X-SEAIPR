import numpy as np 

def heteroscedasticLoss(true, mean, var):
    precision = 1/var
    log_var = np.log(var)    
    loss = (precision * (true - mean)**2 + log_var).mean()
    return loss

def squaredLoss(preds, target):
    loss = ((preds - target) ** 2).sum()
    return loss

def squaredLossExpScale(preds, target):
    eps = 1e-8
    preds = np.log(preds + eps)
    target = np.log(target + eps)
    loss = ((preds - target) ** 2).sum()
    return loss

