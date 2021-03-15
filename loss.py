import numpy as np

def MSELoss(yhat, y):
    s = (np.square(yhat-y))
    s = np.sum(s)/len(y)
    return s

#  MSELoss(np.array([1,2,3]), np.array([1,3,3]))
