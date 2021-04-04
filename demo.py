# # Importing the libraries

# %%
from sklearn import datasets
import numpy as np
import time
import tinydl as dp
from tinydl.layers import *
from tinydl.model import Model
from tinydl.trainer import *

init_time = time.time()


# # Defining the Network

# +
class Net(Model):
    def __init__(self, numClasses):
        super().__init__()
        self.fc1 = Linear(4, 16, activation="relu", name="fc1")
        self.fc2 = Linear(16, 32, activation="relu", name="fc2")
        self.fc3 = Linear(32, 16, activation="relu", name="fc2")
        self.fc4 = Linear(16, numClasses, activation="sigmoid", name="fc3", init="he")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


numClasses = 1
model = Net(numClasses=numClasses)
model.summary()
# -

# # Loading the data

train_X, train_y = datasets.load_iris(return_X_y=True)
X, y = np.asarray(train_X[:100]), np.asarray(train_y[:100])
yi = np.argwhere(y <= 1)
y = np.reshape(y[yi], (-1))
X = np.reshape(X[yi], (y.shape[0], -1))
X = (X - X.min()) / (X.max() - X.min())
X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)


# # Training loop

train(X, y, model)
print(f"Took {(time.time()-init_time)/60} minutes to run")
