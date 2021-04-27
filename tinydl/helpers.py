import numpy as np
import os
from config import *
import pickle
from tinydl.tensor import Tensor

"""
This module has some extra features that would be nice to have but are not strictly deep learning related
"""


def pbar(
    iterable,
    length=50,
    listofextra=[],
    prefix="",
    suffix="",
    decimals=1,
    fill="█",
    printEnd="\r",
):
    """
    Args:
        iterable 
        length (int, optional): . Defaults to 50. Change if you have a big/small screen.
        listofextra (list, optional): . Defaults to []. Extra text
        prefix (str, optional): . Defaults to "".
        suffix (str, optional): . Defaults to "".
        decimals (int, optional): . Defaults to 1.
        fill (str, optional): . Defaults to "█". Change if you want a different block
        printEnd (str, optional): . Defaults to "\r".
    Yields:
        
    Custom progress bar.
    """
    total = len(iterable)
    listofextra = " ".join(listofextra)

    def printPbar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% | {listofextra} {suffix}", end=printEnd)

    printPbar(0)
    for i, item in enumerate(iterable):
        yield item
        printPbar(i + 1)

    print()


def info(arr, n="", p=0):
    """
    Args:
        arr 
        n (str, optional): . Defaults to "".
        p (int, optional): . Defaults to 0.
    Give an array, get a description. Shape, count, mean etc.
    """
    print(f"name : {n}")
    print(f"shape : {arr.shape}")
    print(f"count : {len(arr)}")
    print(f"mean : {np.mean(arr)}")
    print(f"std : {np.std(arr)}")
    if p == 1:
        print(arr)


def checkifdir(logdir=logdir):
    """
    Args:
        logdir . Defaults to logdir.
    Check if the directory exists
    """
    if not os.path.isdir(logdir):
        os.mkdir(logdir)


def getexpno(logdir=logdir):
    """
    Args:
        logdir . Defaults to logdir.
    Returns:
    Return the current experiment number
    """
    return len(os.listdir(logdir))  # Offset by 1 because of .gitkeep


def savemodel(model, total_loss):
    """
    Args:
        model 
        total_loss 
    Save the model to a pickle file
    """
    with open("model.pkl", "wb+") as f:
        pickle.dump(
            {
                "parameters": [x.data for x in model.parameters()],
                "loss": total_loss.data,
            },
            f,
        )


def loadmodel():
    """
    Args:
        None
    Loads the saved model from a pickle file
    """
    with open("model.pkl", "rb+") as f:
        te = pickle.load(f)
        return te

def tensorToArray(arr):
    """
    Args:
        tensor
    Takes an array of Tensors and returns a numpy array
    """
    try:
        return np.array([x.data for x in arr])
    except AttributeError:
        return np.array([x.data for x in np.array(arr)])

def arrayToTensor(arr):
    """
    Args:
        Array
    Takes an array of Tensors and returns a numpy array
    """
    try:
        return [Tensor(x) for x in arr]
    except AttributeError:
        return [Tensor(x) for x in np.array(arr)]

