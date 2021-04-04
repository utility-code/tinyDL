from tinydl.tensor import *
import numpy as np
import tinydl as dp


def MSELoss(yb, ypred):
    _loss = [(yb - ypb) * (yb - ypb) for yb, ypb in zip(yb, ypred)]

    return sum(_loss) * dp.Tensor(1 / len(yb))
    #  s = (np.square(np.subtract(np.array(yb),np.array(ypred))))
    #  return np.sum(s)/len(yb)


def identifyClassFromProb(probs_):
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def accuracy(yhat, y):
    yhat = np.array([float(x.data) for x in yhat])
    y = np.array([float(x.data) for x in y])
    y_hat_ = identifyClassFromProb(yhat)
    return (yhat == y).all(axis=0).mean()
