import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

# pls try to make this autodiff
def sigmoid_backward(dx, x):
    sig = sigmoid(x)
    return dx * sig * (1 - sig)

def relu_backward(dx, x):
    dx = np.array(dx, copy = True)
    dx[x <= 0] = 0;
    return dx;
