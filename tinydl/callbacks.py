from tinydl.config import *
import sys

if usegpu == True:
    import cupy as np
else:
    import numpy as np

def earlystopping(acchistory,after = 3, threshold = .8, patience = .6):
    # For accuracy
    if len(acchistory) > after:
        losse = acchistory[-after:]
        curr = acchistory[-1]
        thr = [x for x in losse if (x-curr)<=threshold]
        patience = len(thr)/len(losse)
        if patience == 1:
            print(f"Early stopping : patience {patience}")
            sys.exit()

        elif patience <= threshold:
            print(f"Early stopping : patience {patience}")
            sys.exit()



    

