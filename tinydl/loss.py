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

def MSELoss(model, yhat, y):
#      yhat, y = sendlosstogpu(yhat, y)
    s = (yhat-y)**2
    s = sum(s)
    return s
#
def SVMLoss(model, yb, scores):
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    return total_loss

def CELoss(yhat, y):
    pass
#
#      yhat, y = sendlosstogpu(yhat, y)
#
#      N = yhat.shape[0]
#      ce = -np.sum(y * np.log(yhat)) / N
#      return ce
#
#  def identifyClassFromProb(probs):
#      if usegpu == True:
#          probs = np.asarray(probs)
#
#      probs_ = np.copy(probs)
#      probs_[probs_ > 0.5] = 1
#      probs_[probs_ <= 0.5] = 0
#      return probs_

