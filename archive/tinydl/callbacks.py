from tinydl.config import *
import sys
import matplotlib.pyplot as plt

if usegpu == True:
    import cupy as np
else:
    import numpy as np


def earlystopping(
    acchistory, losshistory=None, i=None, after=3, threshold=0.8, patience=0.6
):
    # For accuracy
    if len(acchistory) > after:
        losse = acchistory[-after:]
        curr = acchistory[-1]
        thr = [x for x in losse if (x - curr) <= threshold]
        patience = len(thr) / len(losse)
        if patience == 1:
            print(f"Early stopping : patience {patience}")
            sys.exit()

        elif patience <= threshold:
            print(f"Early stopping : patience {patience}")
            sys.exit()


def saveplots(acchistory, losshistory, i):
    if i % afterEvery == 0:
        plt.clf()
        plt.cla()
        plt.plot(losshistory, label="loss")
        plt.plot(acchistory, label="accuracy")
        plt.legend()
        plt.savefig(f"./experiments/epoch_{i}.png")
