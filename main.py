from dataloader import *
from trainer import *
from helpers import *
from layers import *
from config import *


#  class Dataset:
#  def __init__(self, size):
#      self.size = size
#
#  def __len__(self):
#      return self.size
#
#  def __getitem__(self, index):
#          return np.zeros((3, 32, 32)), 1
#  ds = Dataset(1024)
#  print("Created ds")
#  dl = DataLoader(ds, num_workers=4, batch_size=64)
#
#  x, y = next(dl)
#  print(x.shape, y.shape)
#
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
N_SAMPLES = 1000

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

sh1 = y_train.shape[0]
print(X.shape, y.shape)


arch = [
    linear(2, 25, relu),
    linear(25, 50, relu),
    linear(50, 50, relu),
    linear(50, 25, relu),
    linear(25, 1, sigmoid)
]
pretty(arch)

params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((sh1, 1))), arch,epochs=numEpochs, afterEvery=afterEvery, verbose = verbose)

ytesthat, _ = forward(np.transpose(X_test), params_values, arch)
testacc = accuracy(ytesthat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print(f"Validation accuracy : {testacc}")
