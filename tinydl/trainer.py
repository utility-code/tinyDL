from tinydl.layers import *
from tinydl.helpers import *
from tinydl.config import *
from tinydl.loss import *
from tinydl.logger import *
from tinydl.autograd import *
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
        if layer.name in ["linear"]:
            layer_in_size = layer["input_dim"]
            layer_out_size = layer["output_dim"]

            param_values['W' + str(current_idx)] = Tensor(np.random.randn(
                layer_out_size, layer_in_size
            ) * 0.1)
            param_values['b' + str(current_idx)] = Tensor(np.random.randn(
                layer_out_size, 1
            ) * 0.1)

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
            param_values['W' + str(current_idx)] = Tensor(lower + nums*(upper - lower))
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
            param_values['W' + str(current_idx)] = Tensor(lower + nums*(upper - lower))
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
            param_values['W' + str(current_idx)] = Tensor(lower + nums*(upper - lower))
    return param_values


def defaultInit(arch, seed=99):
    # basically just use linear for linear layers and He init for activations
    param_values = {}
    linearInit(arch, param_values=param_values)
    heInit(arch, param_values=param_values)
    normalizedXavierInit(arch, param_values=param_values)
    return param_values

def SGD(model,k):
    new_lr = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= new_lr * p.grad

def ADAM(param_values, gradsVals, arch, lr=0.01):
    pass

dict_optim = {
    "ADAM" : ADAM,
    "SGD" : SGD,
}

dict_loss = {
    "CE" : CELoss,
    "MSE" : MSELoss,
    "SVM" : SVMLoss
}


def forward(x, y, model, bs = None):
    if bs is None:
        xb, yb = x, y
    else:
        rinde = np.random.permutation(x.shape[0])[:bs]
        xb, yb = x[rinde], y[rinde]

    inputs = [list(map(Tensor, xrow)) for xrow in xb]
    out = list(map(model, inputs))

    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, out)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    acc = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, out)]
    total_acc = sum(acc) / len(acc)

    return total_loss , total_acc

def train(x,y, model, epochs = 1, lr= 0.001,bs = batchsize, verbose = True, afterEvery = 10):
    losshistory, acchistory = [],[]
    if log == True:
        checkifdir()
        exp_no = getexpno()
        print(f"Experiment number : {exp_no}")
        exp_file = open(f"{logdir}experiment_{exp_no}.txt", "w+")
        exp_file.write(f"pretty(model)\n\n")
        exp_file.write("epoch,loss,accuracy\n")

    for i in pbar(range(epochs), length=pbarLength):
        total_loss, acc = forward(x,y, model,bs)
        losshistory.append(total_loss)
        acchistory.append(acchistory)
        model.zero_grad()
        total_loss.backward()

        dict_optim[optim](model, i)
        if (log == True and i% logAfter == 0):
            exp_file.write(f"{str(i)}, {str(total_loss)}, {str(acc*100)}%\n")
            exp_file.flush()

        if (i % afterEvery == 0):
            if verbose:
                print(f"Loss : {total_loss.data} , Acc : {acc*100}%")
    if log == True:
        exp_file.close()
    if plotLoss == True:
        plt.plot(losshistory)
    if plotAcc == True:
        plt.plot(acchistory)
    plt.show()



