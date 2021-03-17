from dataloader import *
from trainer import *
from helpers import *
from layers import *
from config import *

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
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


#  from sklearn.datasets import make_moons
#  from sklearn.model_selection import train_test_split
#  N_SAMPLES = 1000
#
#  X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
#
#  sh1 = y_train.shape[0]
#
#  class Dataset:
#      def __init__(self, x,y):
#        self.x = x
#        self.y = y
#        info(self.y)
#
#      def __len__(self):
#        return len(self.x)
#
#      def __getitem__(self, index):
#          return self.x[index], self.y
#
#  ds = Dataset(np.transpose(X_train), y_train.reshape((1, sh1)))
#  print("Created ds")
#  dl = DataLoader(ds, num_workers=12, batch_size=128)
#

#  class Dataset:
#      def __init__(self, size=2048, load_time=0.0005):
#          self.size, self.load_time = size, load_time
#
#      def __len__(self):
#          return self.size
#
#      def __getitem__(self, index):
#          return np.zeros((1, 28, 28)), 1  # return img, label
#
#  ds = Dataset(1024)
#  dl = DataLoader(ds, num_workers=12, batch_size=128)
#
#  arch = [
#      linear(2, 25, relu),
#      linear(25, 50, relu),
#      linear(50, 50, relu),
#      linear(50, 25, relu),
#      linear(25, 1, sigmoid)
#  ]
#  pretty(arch)
#
#  params_values = train(dl, arch,epochs=numEpochs, afterEvery=afterEvery, verbose = verbose)
#
#  ytesthat, _ = forward(np.transpose(X_test), params_values, arch)
#  testacc = accuracy(ytesthat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
#  print(f"Validation accuracy : {testacc}")
