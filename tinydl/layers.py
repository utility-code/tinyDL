from tinydl.helpers import arrayToTensor, tensorToArray
import numpy as np
from .tensor import Tensor
from config import *
import tinydl as dp

class Node:
    """
    Defines a Node class. This contains a linear node
    """

    def __init__(self, n_inputs, activation=None, init="random"):
        self.BAIS_INIT = 0.0
        self.b = Tensor(self.BAIS_INIT)
        self.activation = activation
        self.dict_init = {
            "he": self.heInit,
            "xavier": self.xavierInit,
            "nxavier": self.normalizedXavierInit,
            "random": self.linearInit,
        }
        self.w = self.dict_init[init](n_inputs)

    def __call__(self, x):
        _act = (wi * xi for wi, xi in zip(self.w, x))
        act = sum(_act, self.b)
        if self.activation:
            activation_fxn = getattr(act, self.activation)
            act = activation_fxn()
        return act

    def params(self):
        return self.w + [self.b]

    def dropout(self, arr, p=0.5):
        """

        Args:
            arr 
            p (float, optional): . Defaults to 0.5.

        Returns:
            
        Simple dropout
        """
        return arr * np.random.binomial(1, p, size=arr.shape)

    def linearInit(self, n_inputs):
        """

        Args:
            n_inputs 

        Returns:
            
        Uniform random initialization
        """
        return [Tensor(np.random.uniform(-1, 1)) for _ in range(n_inputs)]

    def xavierInit(self, n_inputs):
        """

        Args:
            n_inputs 

        Returns:
            
        Xavier/He initialization
        """
        lower, upper = -(1.0 / np.sqrt(n_inputs)), (1.0 / (np.sqrt(n_inputs)))
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]

    def normalizedXavierInit(self, n_inputs):
        """

        Args:
            n_inputs 

        Returns:
            
        Normalized Xavier initialization
        """
        lower, upper = -(np.sqrt(6.0) / np.sqrt(n_inputs)), (
            np.sqrt(6.0) / np.sqrt(n_inputs)
        )
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]

    def heInit(self, n_inputs):
        """

        Args:
            n_inputs 

        Returns:
            
        He initialization
        """
        lower, upper = -(1.0 / np.sqrt(n_inputs)), (1.0 / (np.sqrt(n_inputs)))
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]


class Linear:
    """
    Defines the linear layer
    """

    def __init__(self, n_inputs, n_out, name=None, **kwargs):
        self.nodes = [Node(n_inputs, **kwargs) for _ in range(n_out)]
        self.name = name
        self.n_inputs = n_inputs
        self.n_out = n_out

    def __call__(self, x):
        out = [n(x) for n in self.nodes]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.nodes for p in n.params()]

    def summary(self):
        """

        Returns:
            
        Shows a tiny model structure defination
        """
        return {
            "name": self.name,
            "shape": (self.n_inputs, self.n_out),
            "params": len(self.parameters()),
        }

#  def conv2d(image, kernel, padding=0, strides=1):
#      kernel = np.flipud(np.fliplr(kernel))
#
#      xKernShape, yKernShape = kernel.shape[0], kernel.shape[1]
#      xImgShape, yImgShape = image[0], image[1]
#
#      xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
#      yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
#      output = np.zeros((xOutput, yOutput))
#
#      if padding != 0:
#          imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
#          imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
#      else:
#          imagePadded = image
#
#      for y in range(image.shape[1]):
#          if y > image.shape[1] - yKernShape:
#              break
#          if y % strides == 0:
#              for x in range(image.shape[0]):
#                  if x > image.shape[0] - xKernShape:
#                      break
#                  try:
#                      if x % strides == 0:
#                          output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
#                  except:
#                      break
#
#      return output
#
def maxpool(feature_map, size=2, stride=2):
    """
    Args:
        feature_map
        size of pool
        stride size
    Max pool output
    """
    feature_map = tensorToArray(feature_map)
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride),np.uint16((feature_map.shape[1]-size+1)/stride),feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0]-size-1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = np.max(
                    feature_map[r:r+size,  c:c+size])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out

def minpool(feature_map, size=2, stride=2):
    """
    Args:
        feature_map
        size of pool
        stride size

    Min pool output
    """
    feature_map = tensorToArray(feature_map)
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride),np.uint16((feature_map.shape[1]-size+1)/stride),feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0]-size-1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = np.min(
                    feature_map[r:r+size,  c:c+size])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out

def avgpool(feature_map, size=2, stride=2):
    """
    Args:
        feature_map
        size of pool
        stride size

    Avg pool output
    """
    feature_map = tensorToArray(feature_map)
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride),np.uint16((feature_map.shape[1]-size+1)/stride),feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0]-size-1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = np.average(
                    feature_map[r:r+size,  c:c+size])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out
