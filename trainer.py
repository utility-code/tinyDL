import numpy as np
from layers import *
from helpers import pbar
from loss import *

# list of activation functions except sigmoid and tanh
act_layers = ["softplus", "relu", "prelu", "lrelu", "elu", "swish"]


def linearInit(arch, param_values, seed=99):
    # for linear layers
    np.random.seed(seed)

    for idx, layer in enumerate(arch):
        current_idx = idx+1
        if layer["name"] in ["linear"]:
            layer_in_size = layer["input_dim"]
            layer_out_size = layer["output_dim"]

            param_values['W' + str(current_idx)] = np.random.randn(
                layer_out_size, layer_in_size
            ) * 0.1
            param_values['b' + str(current_idx)] = np.random.randn(
                layer_out_size, 1
            ) * 0.1

    return param_values


def xavierInit(arch, param_values, seed=99):
    # for activation functions, mostly for tanh
    np.random.seed(seed)
    for idx, layer in enumerate(arch):
        current_idx = idx+1
        if layer["name"] in ["tanh", "sigmoid"]:
            n = layer["input_dim"]
            lower, upper = -(1.0 / np.sqrt(n)), (1.0/np.sqrt(n))
            nums = np.random.randn(1000)
            param_values['W' + str(current_idx)] = lower + nums*(upper - lower)
    return param_values


def normalizedXavierInit(arch, param_values, seed=99):
    # for activation functions - normalized Xavier
    np.random.seed(seed)
    for idx, layer in enumerate(arch):
        current_idx = idx+1
        if layer["name"] in ["tanh", "sigmoid"]:
            n = layer["input_dim"]
            m = layer["output_dim"]
            lower, upper = -(np.sqrt(6.0) / np.sqrt(n + m)
                             ), (np.sqrt(6.0) / np.sqrt(n + m))
            nums = np.random.randn(1000)
            param_values['W' + str(current_idx)] = lower + nums*(upper - lower)
    return param_values


def heInit(arch, param_values, seed=99):
    # for activation functions, mostly for relu
    np.random.seed(seed)
    for idx, layer in enumerate(arch):
        current_idx = idx+1
        if layer["name"] in act_layers:
            n = layer["input_dim"]
            lower, upper = -(1.0 / np.sqrt(n)), (1.0/np.sqrt(n))
            nums = np.random.randn(1000)
            param_values['W' + str(current_idx)] = lower + nums*(upper - lower)
    return param_values


def defaultInit(arch, seed=99):
    # basically just use linear for linear layers and He init for activations
    param_values = {}
    linearInit(arch, param_values=param_values)
    heInit(arch, param_values=param_values)
    return param_values


def singleForward(a_prev, w_curr, b_curr, activation=relu):
    z_curr = np.dot(w_curr, a_prev) + b_curr
    return activation(z_curr)["value"], z_curr


def forward(x, param_values, arch):
    memory = {}
    a_curr = x
    for idx, layer in enumerate(arch):
        layer_idx = idx + 1
        a_prev = a_curr
        w_curr = param_values["W"+str(layer_idx)]
        b_curr = param_values["b"+str(layer_idx)]
        a_curr, z_curr = singleForward(
            a_prev=a_curr, w_curr=w_curr, b_curr=b_curr)
        memory["A"+str(idx)] = a_prev
        memory["Z"+str(layer_idx)] = z_curr

    return a_curr, memory


def singleBackward(da_curr, w_curr, b_curr, z_curr, a_prev, activation=relu):
    m = a_prev.shape[1]

    # hax. change later
    if activation.__name__ == "relu":
        backact = reluBackward
    elif activation.__name__ == "sigmoid":
        backact = sigmoidBackward
    else:
        backact = reluBackward

    dzcurr = backact(da_curr, z_curr)
    print(dzcurr)
    dwcurr = np.dot(dzcurr, a_prev.T)/m
    dbcurr = np.sum(dzcurr, axis=1, keepdims=True)/m
    daprev = np.dot(w_curr.T, dzcurr)

    return daprev, dwcurr, dbcurr


def backward(yhat, y, memory, param_values, arch):
    gradsVals = {}
    m = y.shape[1]
    y = y.reshape(yhat.shape)

    daprev = - (np.divide(y, yhat) - np.divide(1-y, 1-yhat))

    for layer_idx_prev, layer in reversed(list(enumerate(arch))):
        layer_idx = layer_idx_prev + 1
        activCurr = layer["activation"]

        da_curr = daprev

        aprev = memory["A" + str(layer_idx_prev)]
        z_curr = memory["Z" + str(layer_idx)]

        w_curr = param_values["W" + str(layer_idx)]
        b_curr = param_values["b" + str(layer_idx)]

        da_prev, dw_curr, db_curr = singleBackward(
            da_curr, w_curr, b_curr, z_curr, aprev, activCurr
        )

        gradsVals["dW" + str(layer_idx)] = dw_curr
        gradsVals["db" + str(layer_idx)] = db_curr

    return gradsVals


def update(param_values, gradsVals, arch, lr=0.01):
    for layer_idx, layer in enumerate(arch, 1):
        param_values["W" + str(layer_idx)] -= lr * \
            gradsVals["dW" + str(layer_idx)]
        param_values["b" + str(layer_idx)] -= lr * \
            gradsVals["db" + str(layer_idx)]
    return param_values


def train(x, y, arch, epochs=1, lr=0.01, verbose=True, callback=None, afterEvery=5):
    param_values = defaultInit(arch)
    losshistory = []
    acc_history = []

    for i in pbar(range(epochs)):
        yhat, cache = forward(x, param_values, arch)
        loss = MSELoss(yhat, y)
        losshistory.append(loss)
        acc = accuracy(yhat, y)
        acc_history.append(acc)

        gradsVals = backward(
            yhat, y, cache, param_values, arch
        )
        param_values = update(param_values, gradsVals, arch, lr)

        if (i % afterEvery == 0):
            if verbose:
                print(f"Loss : {loss} , Acc : {acc}")
    return param_values
