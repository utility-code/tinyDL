from tinydl.tensor import *
import numpy as np
import tinydl as dp
from sklearn.metrics import log_loss

"""[summary]
This module has some loss functions and accuracy metrics implemented.
"""


def MSELoss(ypred, yb):
    """[summary]

    Args:
        yb ([type]): [description]
        ypred ([type]): [description]

    Returns:
        [type]: [description]
    Mean squared Error Loss
    """
    _loss = [(yb - ypb) * (yb - ypb) for yb, ypb in zip(yb, ypred)]

    return sum(_loss) * dp.Tensor(1 / len(yb))


def BCELoss(ypred, yb):  # DOESNT WORK YET
    """[summary]

    Args:
        yb ([type]): [description]
        ypred ([type]): [description]

    Returns:
        [type]: [description]
    Binary Cross Entropy Loss
    """
    ce = sum(ypred) * dp.Tensor(1 / len(yb))

    return ce
    #  ypred = [float(x._data) for x in ypred]
    #  yb = [float(x._data) for x in yb]
    #  return dp.Tensor(log_loss(yb, ypred))


def identifyClassFromProb(probs_):
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def accuracy(yhat, y):
    """[summary]

    Args:
        yhat ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    Returns the accuracy
    """
    yhat = np.array([float(x.data) for x in yhat])
    y = np.array([float(x.data) for x in y])
    y_hat_ = identifyClassFromProb(yhat)
    #  return (yhat == y).all(axis=0).mean()
    return (y_hat_ == y).mean()
