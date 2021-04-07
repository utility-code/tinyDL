import numpy as np
from .tensor import Tensor
from config import *


class Node:
    """[summary]
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
        """[summary]

        Args:
            arr ([type]): [description]
            p (float, optional): [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        Simple dropout
        """
        return arr * np.random.binomial(1, p, size=arr.shape)

    def linearInit(self, n_inputs):
        """[summary]

        Args:
            n_inputs ([type]): [description]

        Returns:
            [type]: [description]
        Uniform random initialization
        """
        return [Tensor(np.random.uniform(-1, 1)) for _ in range(n_inputs)]

    def xavierInit(self, n_inputs):
        """[summary]

        Args:
            n_inputs ([type]): [description]

        Returns:
            [type]: [description]
        Xavier/He initialization
        """
        lower, upper = -(1.0 / np.sqrt(n_inputs)), (1.0 / (np.sqrt(n_inputs)))
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]

    def normalizedXavierInit(self, n_inputs):
        """[summary]

        Args:
            n_inputs ([type]): [description]

        Returns:
            [type]: [description]
        Normalized Xavier initialization
        """
        lower, upper = -(np.sqrt(6.0) / np.sqrt(n_inputs)), (
            np.sqrt(6.0) / np.sqrt(n_inputs)
        )
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]

    def heInit(self, n_inputs):
        """[summary]

        Args:
            n_inputs ([type]): [description]

        Returns:
            [type]: [description]
        He initialization
        """
        lower, upper = -(1.0 / np.sqrt(n_inputs)), (1.0 / (np.sqrt(n_inputs)))
        return [
            Tensor(lower + np.random.randn() * (upper - lower)) for _ in range(n_inputs)
        ]


class Linear:
    """[summary]
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
        """[summary]

        Returns:
            [type]: [description]
        Shows a tiny model structure defination
        """
        return {
            "name": self.name,
            "shape": (self.n_inputs, self.n_out),
            "params": len(self.parameters()),
        }
