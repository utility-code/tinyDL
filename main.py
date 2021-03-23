from tinydl.dataloader import *
from tinydl.trainer import *
from tinydl.helpers import *
from tinydl.layers import *
from tinydl.config import *
from tinydl.autograd import *
import time
import random
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

if usegpu == True:
    import cupy as np
    print("Using GPU")
else:
    import numpy as np
    print("No GPU")
np.random.seed(1337)
random.seed(1337)

init_time = time.time()


N_SAMPLES = 100

x, y = make_moons(n_samples = N_SAMPLES, noise=0.1)
y = y*2 -1
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
#
#  sh1 = y_train.shape[0]
#  print(X.shape, y.shape)
#
#  X_train , y_train = np.transpose(X_train), np.transpose(y_train.reshape((sh1, 1)))
#  X_test , y_test = np.transpose(X_test), np.transpose(y_test.reshape((y_test.shape[0], 1)))
#
#  arch = Net(2, [16, 16, 1])

arch = Net(
[
    Layer(2,16),
    Layer(16,16),
    Layer(16,1,nonlin= False),
]
)

pretty(arch)

train(x, y, arch,epochs=numEpochs, afterEvery=afterEvery, verbose = verbose)

print(f"Took {(time.time()-init_time)/60} minutes to run")
