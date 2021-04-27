from config import *
from tinydl.helpers import *
from tinydl.loss import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tinydl as td


def GD(step, model, learning_rate, vw, vw2):
    """

    Args:
        step 
        model 
        learning_rate 
        vw 
        vw2 
    gradient descent optimizer
    """
    for p in model.parameters():
        p._data -= learning_rate * p.grad


def GDM(step, model, learning_rate, vw, vw2):
    """

    Args:
        step 
        model 
        learning_rate 
        vw 
        vw2 
    Gradient descent optimizer with momentum

    """
    for p in model.parameters():
        p._data += momentum * vw - learning_rate * p.grad


def NGD(step, model, learning_rate, vw, vw2):
    """

    Args:
        step 
        model 
        learning_rate 
        vw 
        vw2 
    gradient descent optimizer with Nesterov momentum

    """
    for p in model.parameters():
        p._data += momentum * vw - learning_rate * p.grad


def RMSProp(step, model, learning_rate, vw, vw2):
    """

    Args:
        step 
        model 
        learning_rate 
        vw 
        vw2 
    RMSProp Optimizer
    """
    for p in model.parameters():
        vw = decay * vw + (1.0 - decay) * p.grad ** 2
        p._data -= learning_rate * p.grad / (np.sqrt(vw) + eps)


def ADAM(step, model, learning_rate, vw, vw2):
    """

    Args:
        step 
        model 
        learning_rate 
        vw 
        vw2 
    ADAM Optimizer
    """
    for p in model.parameters():
        vw = beta1 * vw + (1.0 - beta1) * p.grad
        vw2 = beta2 * vw2 + (1.0 - beta2) * p.grad ** 2
        vwUnbiased = vw / (1.0 - beta1 ** (step + 1))
        vw2Unbiased = vw2 / (1.0 - beta2 ** (step + 1))
        p._data -= learning_rate * vwUnbiased / (np.sqrt(vw2Unbiased) + eps)


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
    """

    Args:
        X 
        y 
        model 
    Main training loop. Contains logging functions, somewhat of a mini batcher and loss/accuracy plotter.
    """
    lossHistory = []

    if log == True:
        checkifdir()
        exp_no = getexpno()
        print(f"Experiment number : {exp_no}")
        exp_file = open(f"{logdir}experiment_{exp_no}.txt", "w+")
        currtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        exp_file.write(f"Experiment conducted on : {currtime}\n")
        exp_file.write("epoch,loss\n")

    if (resumetrain == True) and (modelpath != None):
        savedmo = loadmodel()

    for steps in pbar(range(numEpochs)):
        ri = np.random.permutation(X.shape[0])[:batch_size]
        xb = X[ri]
        yb = y[ri]
        xb = [list(map(td.Tensor, x)) for x in xb]
        vw = 0.0
        vw2 = 0.0
        # forward
        y_pred_b = list(map(model.forward, xb))
        yb = [td.Tensor(y) for y in yb]
        #  total_loss = dict_loss[lossFunction](y_pred_b, yb)
        total_loss = getattr(td.loss, lossFunction)(y_pred_b, yb)
        lossHistory.append(total_loss.data)

        # accuracy
        print("\nTrain Accuracy : ", getattr(td.loss, accuracy_metric)(yb, y_pred_b), "%")
        # ADD CHECK FOR EMPTY FILE
        if (steps == 0) and (resumetrain == True) and (modelpath != None):
            savedmo = loadmodel()
            total_loss = savedmo["loss"]
            print(model.parameters)
            model.parameters = savedmo["parameters"]  # doesnt work

        # backward
        model.init_backward()
        total_loss.backward()

        if log == True and steps % afterEvery == 0:
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
