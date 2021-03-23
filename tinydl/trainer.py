from tinydl.layers import *
from tinydl.helpers import *
from tinydl.config import *
from tinydl.loss import *
from tinydl.callbacks import *
import matplotlib.pyplot as plt

if usegpu == True:
    import cupy as np
else:
    import numpy as np
# list of activation functions except sigmoid and tanh
act_layers = ["softplus", "relu", "prelu", "lrelu", "elu", "swish"]

# optimizer
def SGD(model,k):
    new_lr = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= new_lr * p.grad

def GD(model,lr=lr):
    for p in model.parameters():
        p.data -= lr * p.grad

def ADAM(param_values, gradsVals, arch, lr=0.01):
    pass

dict_optim = {
    "ADAM" : ADAM,
    "SGD" : SGD,
    "GD" : GD,
}

dict_loss = {
    "CE" : CELoss,
    "MSE" : MSELoss,
    "SVM" : SVMLoss
}


def forward(x, y, model, bs = None):
    x = cuda(x)
    y = cuda(y)
    if bs is None:
        xb, yb = x, y
    else:
        rinde = np.random.permutation(x.shape[0])[:bs]
        xb, yb = x[rinde], y[rinde]

    inputs = [list(map(Tensor, xrow)) for xrow in xb]
    if layerdropout == True:
        for layer in model.layers:
            dropout(layer.parameters(), layerdropoutprob)
    out = list(map(model, inputs))

    total_loss = dict_loss[lossfunc](model, yb, out)
    
    acc = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, out)]
    total_acc = sum(acc) / len(acc)

    return total_loss , total_acc

def train(x,y, model, epochs = 1, lr= 0.001,bs = batchsize, verbose = True, afterEvery = 10, callbacks = []):
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
        losshistory.append(total_loss.data)
        acchistory.append(acc)
        model.zero_grad()
        total_loss.backward()

        dict_optim[optim](model, i)
        if (log == True and i% logAfter == 0):
            exp_file.write(f"{str(i)}, {str(total_loss)}, {str(acc*100)}%\n")
            exp_file.flush()

        if (i % afterEvery == 0):
            if verbose:
                print(f"Loss : {total_loss.data} , Acc : {acc*100}%")

        for cbs in callbacks:
            cbs(acchistory)
    if log == True:
        exp_file.close()
    if plotLoss == True:
        plt.plot(losshistory)
    if plotAcc == True:
        plt.plot(acchistory)
    plt.show()

