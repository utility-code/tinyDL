import os
from tinydl.config import *

def checkifdir(logdir = logdir):
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

def getexpno(logdir = logdir):
    return len(os.listdir(logdir)) # Offset by 1 because of .gitkeep
