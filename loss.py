import numpy as np

def MSELoss(yhat, y):
    s = (np.square(yhat-y))
    s = np.sum(s)/len(y)
    return s

#  MSELoss(np.array([1,2,3]), np.array([1,3,3]))

def identifyClassFromProb(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def accuracy(yhat, y):
    y_hat_ = identifyClassFromProb(yhat)
    return (y_hat_ == y).all(axis = 0).mean()
