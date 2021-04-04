import sys
from tinydl.config import *
from tensor import Tensor
import numpy as np

# Layers


class Node:
    BAIS_INIT = 0.0

    def __init__(self, n_inputs, activation=None):
        self.w = [Tensor(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Tensor(self.BAIS_INIT)
        self.activation = activation

    def __call__(self, x):
        _act = (wi * xi for wi, xi in zip(self.w, x))
        act = sum(_act, self.b)
        if self.activation:
            activation_fxn = getattr(act, self.activation)
            act = activation_fxn()
        return act

    def params(self):
        return self.w + [self.b]


class Dense:
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
        return {
            "name": self.name,
            "shape": (self.n_inputs, self.n_out),
            "params": len(self.parameters()),
        }


# ACTIVATIONS


def sigmoid(x):
    return {"value": 1 / (1 + np.exp(-x)), "name": "sigmoid"}


def relu(x):
    return {"value": np.maximum(0, x) + 1e-12, "name": "relu"}


def prelu(x, a):  # PRelu
    return {"value": np.maximum(0, x * a), "name": "prelu"}


def lrelu(x, alpha):  # leaky relu
    return {"value": np.maximum(alpha * x, x), "name": "lrelu"}


def softplus(x):
    return {"value": np.log(np.exp(x) + 1), "name": "softplus"}


def elu(x, a):
    return {"value": np.maximum(x, a * (np.exp(x) - 1)), "name": "elu"}


def swish(x, beta):
    return {"value": x / (1 + np.exp(-beta * x)), "name": "swish"}


def tanh(x):
    return {
        "value": (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
        "name": "tanh",
    }


def softmax(x):
    eX = np.exp(x - np.max(x))
    return {"value": eX / eX.sum(axis=0), "name": "softmax"}


# LAYERS


#  def linear(input_dim, output_dim, activation=relu):
#      return {"input_dim": input_dim, "output_dim": output_dim, "name": "linear", "activation": activation}

#  def conv1d(image, kernel, pad = "same"):
#      return np.convolve(image, kernel, mode= pad)
#
#  def conv2d(image, kernel, pad=2, stride=2):
#      # cross correlation, flip horizontally then vertically
#      kernel = np.flipud(np.fliplr(kernel))
#      xkshape, ykshape = kernel.shape
#      ximgshape, yimgshape = image.shape
#
#      xout = int(((ximgshape - xkshape + 2 * pad)/stride)+1)
#      yout = int(((yimgshape - ykshape + 2 * pad)/stride)+1)
#      output = np.zeros((xout, yout))
#
#      if pad != 0:
#          imagepadded = np.zeros(
#              (image.shape[0] + pad*2, image.shape[1] + pad*2))
#          imagepadded[int(pad):int(-1*pad), int(pad):int(-1*pad)] = image
#      else:
#          imagepadded = image
#
#      for y in range(image.shape[1]):
#          if y > image.shape[1] - ykshape:
#              break
#          if y % stride == 0:
#              for x in range(image.shape[0]):
#                  if x > image.shape[0] - xkshape:
#                      break
#                  try:
#                      if x % stride == 0:
#                          output[x, y] = (
#                              kernel*imagepadded[x:x+xkshape, y: y + ykshape]).sum()
#                  except:
#                      break
#
#      return {"value":output, "name":"conv2d"}
#
#  # Backwards : pls try to make this autodiff
#
#
#  def sigmoidBackward(dx, x):
#      sig = sigmoid(x)["value"]
#      return dx * sig * (1 - sig)
#
#
#  def reluBackward(dx, x):
#      dx = np.array(dx, copy=True)
#      dx[x <= 0] = 0
#      return dx
#
#
#  def lreluBackward(x, alpha):
#      return 1 if x > 0 else alpha
#
#
#  def tanhBackward(x):
#      return 1 - np.power(tanh(x)["value"], 2)
