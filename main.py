from dataloader import *
from trainer import *
import numpy as np


class Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        #  return np.zeros((3, 32, 32)), 1
        a = [0, 0, 1, 1, 0, 0,
             0, 1, 0, 0, 1, 0,
             1, 1, 1, 1, 1, 1,
             1, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 1]
        b = [0, 1, 1, 1, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 1, 1, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 1, 1, 1, 0]
        c = [0, 1, 1, 1, 1, 0,
             0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 0]

        y = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        x = np.array([np.array(a).reshape(1, 30), np.array(b).reshape(1, 30),
                      np.array(c).reshape(1, 30)])
        y = np.array(y)
        return x, y


ds = Dataset(1024)
print("Created ds")
dl = DataLoader(ds, num_workers=4, batch_size=64)

x, y = next(dl)
print(x.shape, y.shape)
