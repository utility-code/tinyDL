import numpy as np
import os
from config import *
import pickle

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
    """[summary]
    Args:
        iterable ([type]): [description]
        length (int, optional): [description]. Defaults to 50. Change if you have a big/small screen.
        listofextra (list, optional): [description]. Defaults to []. Extra text
        prefix (str, optional): [description]. Defaults to "".
        suffix (str, optional): [description]. Defaults to "".
        decimals (int, optional): [description]. Defaults to 1.
        fill (str, optional): [description]. Defaults to "█". Change if you want a different block
        printEnd (str, optional): [description]. Defaults to "\r".
    Yields:
        [type]: [description]
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
    """[summary]
    Args:
        arr ([type]): [description]
        n (str, optional): [description]. Defaults to "".
        p (int, optional): [description]. Defaults to 0.
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
    """[summary]
    Args:
        logdir ([type], optional): [description]. Defaults to logdir.
    Check if the directory exists
    """
    if not os.path.isdir(logdir):
        os.mkdir(logdir)


def getexpno(logdir=logdir):
    """[summary]
    Args:
        logdir ([type], optional): [description]. Defaults to logdir.
    Returns:
        [type]: [description]
    Return the current experiment number
    """
    return len(os.listdir(logdir))  # Offset by 1 because of .gitkeep


def savemodel(model, total_loss):
    """[summary]
    Args:
        model ([type]): [description]
        total_loss ([type]): [description]
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
