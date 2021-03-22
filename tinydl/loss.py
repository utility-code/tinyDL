from tinydl.config import *
if usegpu == True:
    import cupy as np
else:
    import numpy as np

def sendlosstogpu(yhat, y):
    if usegpu == True:
        return np.asarray(yhat), np.asarray(y)
    else:
        return yhat, y

def MSELoss(yhat, y):
    yhat, y = sendlosstogpu(yhat, y)
    s = (np.square(yhat-y))
    s = np.sum(s)/len(y)
    return s

#  MSELoss(np.array([1,2,3]), np.array([1,3,3]))

def CELoss(yhat, y):

    yhat, y = sendlosstogpu(yhat, y)

    N = yhat.shape[0]
    ce = -np.sum(y * np.log(yhat)) / N
    return ce

def identifyClassFromProb(probs):
    if usegpu == True:
        probs = np.asarray(probs)

    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def accuracy(yhat, y):
    yhat, y = sendlosstogpu(yhat, y)
    y_hat_ = identifyClassFromProb(yhat)
    return (y_hat_ == y).all(axis = 0).mean()
