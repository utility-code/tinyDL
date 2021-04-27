from tinydl.tensor import *
import numpy as np
import tinydl as dp
from tinydl.helpers import arrayToTensor, tensorToArray
import pysnooper as snp

"""
This module has some loss functions and accuracy metrics implemented.
"""


def MSELoss(yb, ypred):
    """

    Args:
        yb
        ypred

    Returns:

    Mean squared Error Loss
    """
    try:
        _loss = [(yb - ypb) * (yb - ypb) for yb, ypb in zip(yb, ypred)]
    except AttributeError:
        ypred = tensorToArray(ypred)
        yb = arrayToTensor(yb)
        _loss = [(yb - ypb) * (yb - ypb) for yb, ypb in zip(yb, ypred)]

    return sum(_loss) * dp.Tensor(1 / len(yb))


def MAELoss(yb, ypred):  # WIP
    """

    Args:
        yb
        ypred

    Returns:

    Mean Absolute Error Loss
    """
    yb = tensorToArray(yb)
    ypred = tensorToArray(ypred)
    diff = arrayToTensor(np.abs(yb - ypred))
    return sum(diff) * dp.Tensor(1 / len(yb))


def CrossEntropyLoss(yb, ypred):  # WIP
    """

    Args:
        yb
        ypred

    Returns:

    Cross Entropy Loss
    """
    #  yb = tensorToArray(yb)
    #  ypred = tensorToArray(ypred)
    eps = np.finfo(float).eps

    ce = yb * (np.log(ypred + eps))
    return dp.Tensor(-np.sum(ce))


def MeanLogCoshLoss(yb, ypred):  # WIP
    """

    Args:
        yb
        ypred

    Returns:

    Mean Absolute Error Loss
    """
    yb = tensorToArray(yb)
    ypred = tensorToArray(ypred)
    diff = np.log(np.cosh(yb, ypred))
    return dp.Tensor(np.mean(diff))


def MeanHellingerLoss(yb, ypred):  # WIP
    """

    Args:
        yb
        ypred

    Returns:

    Mean Hellinger Loss
    """
    yb = tensorToArray(yb)
    ypred = tensorToArray(ypred)
    diff = np.log(np.cosh(yb, ypred))
    return dp.Tensor(np.mean(diff))


def MeanIOUScore(yb, ypred):  # WIP
    """

    Args:
        yb
        ypred

    Mean IOU Loss
    """
    yb = tensorToArray(yb)
    ypred = tensorToArray(ypred)

    uniqueLabels = set(yb.ravel())
    numUnique = len(uniqueLabels)
    I = np.empty(shape=(numUnique,), dtype=float)
    U = np.empty(shape=(numUnique,), dtype=float)

    for i, val in enumerate(uniqueLabels):
        predI = ypred == val
        lblI = yb == val

        I[i] = np.sum(np.logical_and(lblI, predI))
        U[i] = np.sum(np.logical_or(lblI, predI))

    return dp.Tensor(np.mean(I / U))


def identifyClassFromProb(probs_):
    try:
        probs_[probs_ > 0.5] = 1
    except TypeError:
        probs_ = np.array([x.data for x in probs_])
        probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def blank(yhat, y):
    return dp.Tensor(0.0)


def accuracy(yb, ypred):
    """

    Args:
        yhat
        y

    Returns the accuracy
    """

    ypred = tensorToArray(ypred)
    y_hat_ = identifyClassFromProb(ypred)
    yb = tensorToArray(yb)
    if yb.shape[-1] == 3:
        pass
    elif len(yb.shape) == 1:
        return np.equal(y_hat_, yb).sum()
