from tinydl.dataloader import *
from tinydl.trainer import *
from tinydl.helpers import *
from tinydl.layers import *
from tinydl.config import *
import time
from sklearn.datasets import make_moons

if usegpu == True:
    import cupy as np
    print("Using GPU")
else:
    import numpy as np
    print("No GPU")

init_time = time.time()


N_SAMPLES = 100

x, y = make_moons(n_samples = N_SAMPLES, noise=0.1)
y = y*2 -1

arch = Net(
[
    Layer(2,16,'relu'),
    Layer(16,16,'relu'),
    Layer(16,1),
]
)

pretty(arch)

train(x, y, arch,epochs=numEpochs, afterEvery=afterEvery, verbose = verbose)

print(f"Took {(time.time()-init_time)/60} minutes to run")
