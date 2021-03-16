import numpy as np


def sigmoid(x):
    return {"value": 1/(1+np.exp(-x)), "name": "sigmoid"}


def relu(x):
    return {"value": np.maximum(0, x), "name": "relu"}


def prelu(x, a):  # PRelu
    return {"value": np.maximum(0, x*a), "name": "prelu"}


def lrelu(x, alpha):  # leaky relu
    return {"value": np.maximum(alpha*x, x), "name": "lrelu"}


def softplus(x):
    return {"value": np.log(np.exp(x) + 1), "name": "softplus"}


def elu(x, a):
    return {"value": np.maximum(x, a*(np.exp(x)-1)), "name": "elu"}


def swish(x, beta):
    return {"value": x/(1+np.exp(-beta*x)), "name": "swish"}


def tanh(x):
    return {"value": (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), "name": "tanh"}


def softmax(x):
    eX = np.exp(x - np.max(x))
    return {"value": eX/eX.sum(axis=0), "name": "softmax"}


def linear(input_dim, output_dim, activation = relu):
    return {"input_dim": input_dim, "output_dim": output_dim, "name": "linear", "activation":activation}

# pls try to make this autodiff


def sigmoidBackward(dx, x):
    sig = sigmoid(x)["value"]
    return dx * sig * (1 - sig)


def reluBackward(dx, x):
    dx = np.array(dx, copy=True)
    dx[x <= 0] = 0
    return dx


def lreluBackward(x, alpha):
    return 1 if x > 0 else alpha


def tanhBackward(x):
    return 1 - np.power(tanh(x)["value"], 2)
