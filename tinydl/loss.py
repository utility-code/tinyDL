from tinydl.config import *
if usegpu == True:
    import cupy as np
else:
    import numpy as np

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

def CELoss(model, yhat, y):

    N = yhat.shape[0]
    ce = -sum(y * np.log(yhat)) / N
    return ce

