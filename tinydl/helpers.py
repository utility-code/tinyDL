import numpy as np
import os
from config import *
import pickle

def pbar(iterable, length=50, listofextra=[], prefix='', suffix='', decimals=1, fill='█', printEnd="\r"):
    total = len(iterable)
    listofextra = " ".join(listofextra)

    def printPbar(iteration):
        percent = ("{0:." + str(decimals)+"f}").format(100 *
                                                       (iteration/float(total)))
        filledLength = int(length*iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(
            f'\r{prefix} |{bar}| {percent}% | {listofextra} {suffix}', end=printEnd)

    printPbar(0)
    for i, item in enumerate(iterable):
        yield item
        printPbar(i+1)

    print()


def info(arr, n="", p=0):
    # get some info about an array for debugging
    print(f"name : {n}")
    print(f"shape : {arr.shape}")
    print(f"count : {len(arr)}")
    print(f"mean : {np.mean(arr)}")
    print(f"std : {np.std(arr)}")
    if p == 1:
        print(arr)

def checkifdir(logdir = logdir):
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

def getexpno(logdir = logdir):
    return len(os.listdir(logdir)) # Offset by 1 because of .gitkeep

def savemodel(model, total_loss):
    with open("model.pkl", "wb+") as f:
        pickle.dump({
            "parameters": [x.data for x in model.parameters()],
            "loss": total_loss.data,
        }, f)
