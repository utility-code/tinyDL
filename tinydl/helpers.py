from tinydl.config import *
import os
if usegpu == True:
    import cupy as np
else:
    import numpy as np



def pbar(iterable, length=50,listofextra = [], prefix='', suffix='', decimals=1, fill='█', printEnd="\r"):
    total = len(iterable)
    listofextra = " ".join(listofextra)

    def printPbar(iteration):
        percent = ("{0:." + str(decimals)+"f}").format(100 *
                                                       (iteration/float(total)))
        filledLength = int(length*iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% | {listofextra} {suffix}', end=printEnd)

    printPbar(0)
    for i, item in enumerate(iterable):
        yield item
        printPbar(i+1)

    print()

def pretty(arch):
    print(f"No of layers: {len(arch.layers)}")
    print(f"Number of parameters : {len(arch.parameters())}")
    print(f"Batch size : {batchsize}")
    print(arch)

def info(arr, n = "", p = 0):
    # get some info about an array for debugging
    print(f"name : {n}")
    print(f"shape : {arr.shape}")
    print(f"count : {len(arr)}")
    print(f"mean : {np.mean(arr)}")
    print(f"std : {np.std(arr)}")
    if p==1:
        print(arr)


def checkifdir(logdir = logdir):
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

def getexpno(logdir = logdir):
    return len(os.listdir(logdir)) # Offset by 1 because of .gitkeep
