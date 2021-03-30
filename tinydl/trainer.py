from config import *
from tinydl.helpers import *
from tinydl.loss import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tinydl as dp


def GD(step, model, learning_rate, vw, vw2):
    for p in model.parameters():
        p._data -= learning_rate * p.grad


def GDM(step, model, learning_rate, vw, vw2):
    for p in model.parameters():
        p._data += (momentum*vw - learning_rate * p.grad)


def NGD(step, model, learning_rate, vw, vw2):
    for p in model.parameters():
        p._data += (momentum*vw - learning_rate * p.grad)


def RMSProp(step, model, learning_rate, vw, vw2):
    for p in model.parameters():
        vw = decay*vw + (1. - decay) * p.grad**2
        p._data -= (learning_rate * p.grad / (np.sqrt(vw) + eps))


def ADAM(step, model, learning_rate, vw, vw2):
    for p in model.parameters():
        vw = beta1 * vw + (1. - beta1) * p.grad
        vw2 = beta2*vw2 + (1. - beta2) * p.grad**2
        vwUnbiased = vw / (1. - beta1**(step+1))
        vw2Unbiased = vw2 / (1. - beta2**(step+1))
        p._data -= (learning_rate * vwUnbiased / (np.sqrt(vw2Unbiased) + eps))


dict_optim = {
    "GD": GD,  # Gradient Descent
    "GDM": GDM,  # GD with momentum
    "NGD": NGD,  # GD with nesterov mometum
    "RMSProp": RMSProp,  # RMS Prop
    "ADAM": ADAM,  # RMS Prop
}

dict_loss = {
    "MSE": MSELoss,
}


def train(X, y, model):
    lossHistory = []

    if log == True:
        checkifdir()
        exp_no = getexpno()
        print(f"Experiment number : {exp_no}")
        exp_file = open(f"{logdir}experiment_{exp_no}.txt", "w+")
        currtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        exp_file.write(f"Experiment conducted on : {currtime}\n")
        exp_file.write("epoch,loss\n")

    for steps in pbar(range(numEpochs)):
        ri = np.random.permutation(X.shape[0])[:batch_size]
        xb = X[ri]
        yb = y[ri]
        xb = [list(map(dp.Tensor, x)) for x in xb]
        vw = 0.0
        vw2 = 0.0
        # forward
        y_pred_b = list(map(model.forward, xb))
        yb = [dp.Tensor(y) for y in yb]
        total_loss = dict_loss[lossFunction](y_pred_b, yb)
        lossHistory.append(total_loss.data)
        #  total_acc = accuracy(yb, y_pred_b)
        # backward
        model.init_backward()
        total_loss.backward()

        if (log == True and steps % afterEvery == 0):
            #  print(accuracy(y_pred_b, yb))
            exp_file.write(f"{str(steps)}, {str(total_loss.data)}\n")
            exp_file.flush()

        # Optimize
        dict_optim[optimizer](steps, model, learning_rate, vw, vw2)
        if steps % afterEvery == 0:
            savemodel(model, total_loss)
            print(f"\nloss {total_loss.data} ")

    if log == True:
        exp_file.close()
    if plotLoss == True:
        plt.cla()
        plt.clf()
        plt.plot(lossHistory)
        plt.show()

