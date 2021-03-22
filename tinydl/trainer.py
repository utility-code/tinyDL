from tinydl.layers import *
from tinydl.helpers import *
from tinydl.config import *
from tinydl.loss import *
from tinydl.logger import *
import matplotlib.pyplot as plt

if usegpu == True:
    import cupy as np
else:
    import numpy as np
# list of activation functions except sigmoid and tanh
act_layers = ["softplus", "relu", "prelu", "lrelu", "elu", "swish"]

def dropout(arr, p = 0.5):
    return arr* np.random.binomial(1, p, size = arr.shape)

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
    normalizedXavierInit(arch, param_values=param_values)
    return param_values


def singleForward(a_prev, w_curr, b_curr, idx_break , activation=relu):
    if usegpu== True:
        a_prev = np.asarray(a_prev)
    z_curr = np.dot(w_curr, a_prev) + b_curr
    if layerdropout == True and idx_break!=True:
        z_curr = dropout(z_curr, p = layerdropoutprob)
    if activationdropout == True and idx_break != True:
        return dropout(activation(z_curr)["value"], p = actdropoutprob), z_curr
    else:
        return activation(z_curr)["value"], z_curr


def forward(x, param_values, arch):
    memory = {}
    a_curr = x
    for idx, layer in enumerate(arch):
        layer_idx = idx + 1
        a_prev = a_curr
        w_curr = param_values["W"+str(layer_idx)]
        b_curr = param_values["b"+str(layer_idx)]
        if idx == len(arch) - 1:
            idx_break = True
        else:
            idx_break = False
        a_curr, z_curr = singleForward(
            a_prev=a_curr, w_curr=w_curr, b_curr=b_curr, idx_break= idx_break)
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
    if usegpu == True:
        a_prev = np.asarray(a_prev)
    dwcurr = np.dot(dzcurr, a_prev.T)/m
    dbcurr = np.sum(dzcurr, axis=1, keepdims=True)/m
    daprev = np.dot(w_curr.T, dzcurr)

    return daprev, dwcurr, dbcurr


def backward(yhat, y, memory, param_values, arch):
    gradsVals = {}
    #  y = y.reshape(yhat.shape)
    if usegpu == True:
            yhat = np.asarray(yhat)
            y = np.asarray(y)

    daprev = - (np.divide(y, yhat) - np.divide(1-y, 1-yhat))

    for layer_idx_prev, layer in reversed(list(enumerate(arch))):
        layer_idx = layer_idx_prev + 1
        activCurr = layer["activation"]

        da_curr = daprev

        aprev = memory["A" + str(layer_idx_prev)]
        z_curr = memory["Z" + str(layer_idx)]

        w_curr = param_values["W" + str(layer_idx)]
        b_curr = param_values["b" + str(layer_idx)]

        daprev, dw_curr, db_curr = singleBackward(
            da_curr, w_curr, b_curr, z_curr, aprev, activCurr
        )

        gradsVals["dW" + str(layer_idx)] = dw_curr
        gradsVals["db" + str(layer_idx)] = db_curr

    return gradsVals


def GD(param_values, gradsVals, arch, lr=0.01):
    for layer_idx, layer in enumerate(arch, 1):
        param_values["W" + str(layer_idx)] -= lr * \
            gradsVals["dW" + str(layer_idx)]
        param_values["b" + str(layer_idx)] -= lr * \
            gradsVals["db" + str(layer_idx)]
    return param_values

def SGD(param_values, gradsVals, arch, lr=0.01):
    pass

def ADAM(param_values, gradsVals, arch, lr=0.01):
    pass

dict_optim = {
    "GD" : GD,
    "ADAM" : ADAM,
    "SGD" : SGD,
}

dict_loss = {
    "CE" : CELoss,
    "MSE" : MSELoss,
}


def train(x,y, arch, epochs=1, lr=0.01, verbose=True, callback=None, afterEvery=10):
    param_values = defaultInit(arch)
    losshistory = []
    acc_history = []

    if log == True:
        checkifdir()
        exp_no = getexpno()
        print(f"Experiment number : {exp_no}")
        exp_file = open(f"{logdir}experiment_{exp_no}.txt", "w+")
        exp_file.write(f"Num layers: {len(arch)}\nModel: {pretty(arch)}\n\n")
        exp_file.write("epoch,loss,accuracy\n")

    for i in pbar(range(epochs), length=pbarLength):
        yhat, cache = forward(x, param_values, arch)
        loss = dict_loss[lossfunc](yhat, y)
        losshistory.append(loss)
        acc = accuracy(yhat, y)
        acc_history.append(acc)

        gradsVals = backward(
            yhat, y, cache, param_values, arch
        )
        param_values = dict_optim[optim](param_values, gradsVals, arch, lr)

        if (log == True and i% logAfter == 0):
            exp_file.write(f"{str(i)}, {str(loss)}, {str(acc)}\n")
            exp_file.flush()

        if (i % afterEvery == 0):
            if verbose:
                print(f"Loss : {loss} , Acc : {acc}")
    if log == True:
        exp_file.close()
    if plotLoss == True:
        plt.plot(losshistory)
    if plotAcc == True:
        plt.plot(acc_history)
    plt.show()
    return param_values


