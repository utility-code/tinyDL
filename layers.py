import numpy as np

def sigmoid(x , dA = None):
    # Sigmoid function
    f = (1/(1+ np.exp(-x)))
    {"forward": f, "backward": dA * f}

def relu(z):
    # relu
    return np.maximum(0,z)

