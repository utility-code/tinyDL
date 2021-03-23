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

# Training part

# Model
model =[
    Layer(2,32,'relu'),
    Layer(32,16,'relu'),
    Layer(16,1),
]

arch = Net(model)
pretty(arch)

train(x, y, arch,epochs=numEpochs, 
      afterEvery=afterEvery, verbose = verbose,
     callbacks = [earlystopping])

print(f"Took {(time.time()-init_time)/60} minutes to run")
