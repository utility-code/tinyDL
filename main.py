from tinydl.dataloader import *
from tinydl.trainer import *
from tinydl.helpers import *
from tinydl.layers import *
from tinydl.config import *
import time

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

if usegpu == True:
    import cupy as np
    print("Using GPU")
else:
    import numpy as np
    print("No GPU")

init_time = time.time()
N_SAMPLES = 1000

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

sh1 = y_train.shape[0]
print(X.shape, y.shape)

X_train , y_train = np.transpose(X_train), np.transpose(y_train.reshape((sh1, 1)))
X_test , y_test = np.transpose(X_test), np.transpose(y_test.reshape((y_test.shape[0], 1)))

arch = [
    linear(2, 25, relu),
    linear(25, 50, relu),
    linear(50, 50, relu),
    linear(50, 25, relu),
    linear(25, 1, sigmoid)
]
pretty(arch)

params_values = train(X_train, y_train, arch,epochs=numEpochs, afterEvery=afterEvery, verbose = verbose)

ytesthat, _ = forward(X_test, params_values, arch)
testacc = accuracy(ytesthat, np.transpose(y_test))
print(f"Validation accuracy : {testacc}")

print(f"Took {(time.time()-init_time)/60} minutes to run")
