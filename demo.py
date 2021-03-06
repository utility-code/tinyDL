# %%
# Importing the libraries
import time
from tinydl.layers import *
from tinydl.model import Model
from tinydl.data import *
from tinydl.trainer import *
from tinydl.augmentation import *

# temp
import tinydl as dp
import numpy as np

init_time = time.time()

# %%
# Defining the Network

# %%
# Only linear model
#  class Net(Model):
#      def __init__(self, numClasses):
#          super().__init__()
#          self.fc1 = Linear(13, 16, activation="relu", name="fc1")
#          self.fc2 = Linear(16, 32, activation="relu", name="fc2")
#          self.fc3 = Linear(32, 16, activation="relu", name="fc2")
#          self.fc4 = Linear(16, numClasses, activation="sigmoid",
#                            name="fc3", init="he")
#
#      def forward(self, x):
#          x = self.fc1(x)
#          x = self.fc2(x)
#          x = self.fc3(x)
#          x = self.fc4(x)
#          return x
#

# %% Linear model with pool and flatten
class Net(Model):
    def __init__(self, numClasses):
        super().__init__()
        self.fc1 = Linear(13, 64, activation="relu", name="fc1")
        self.fc4 = Linear(840, numClasses, activation="sigmoid", name="fc3", init="he")

    def forward(self, x):
        x = self.fc1(x)
        x = avgpool(x, 3, 3)
        x = flatten(x)
        x = self.fc4(x)
        return x


numClasses = 1
model = Net(numClasses=numClasses)
model.summary()

# %%
# Loading the data manually from sklearn
#  from sklearn import datasets
#  train_X, train_y = datasets.load_iris(return_X_y=True)
#  X, y = np.asarray(train_X[:100]), np.asarray(train_y[:100])
#  yi = np.argwhere(y <= 1)
#  y = np.reshape(y[yi], (-1))
#  X = np.reshape(X[yi], (y.shape[0], -1))
#  X = (X - X.min()) / (X.max() - X.min())
#  X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)

# %%
# Loading an external dataframe using helpers

# %%
#  fpath = "/media/hdd/Datasets/heart.csv"
#
#  trainX, trainy, testX, testy = DataFrameClassification(
#      fpath, label_col="target", max_rows=100
#  ).read_data()
#
# %%
# Loading an image folder
fpath = "/media/hdd/Datasets/bw2color_subset"

trainX, trainy, testX, testy = ImageFolderClassification(
    fpath=fpath, aug=[Normalize]
).read_data()
#
# %%
# Training loop
train(trainX, trainy, model)

# %%
print(f"Took {(time.time()-init_time)/60} minutes to run")
