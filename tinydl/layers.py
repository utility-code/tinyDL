import sys
from tinydl.config import *
if usegpu == True:
    import cupy as np
else:
    import numpy as np

def randomInit(nin):
    return [Tensor(np.random.randn()*0.1) for _ in range(nin)]

def uniforminit(nin):
    return [Tensor(np.random.uniform(-1,1)) for _ in range(nin)]

def xavierinit(nin):
    lower, upper = -(1.0 / np.sqrt(nin)), (1.0/np.sqrt(nin))
    return [Tensor(lower + np.random.randn(1000)*(upper - lower)) for _ in range(nin)]

def normalizedxavierinit(nin):
    lower, upper = -(np.sqrt(6.0) / np.sqrt(nin)
                             ), (np.sqrt(6.0) / np.sqrt(nin))
    return [Tensor(lower + np.random.randn(1000)*(upper - lower)) for _ in range(nin)]


def dropout(arr, p = 0.5):
    arr = np.array(arr)
    arr= arr* np.random.binomial(1, p, size = arr.shape)

class Tensor:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __exp__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'e{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin='relu'):
        #  self.w = [Tensor(np.random.uniform(-1,1)) for _ in range(nin)]
        self.w = uniforminit(nin)
        self.b = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        dict_o = {
            'relu' : act.relu(),
        }
        return dict_o[self.nonlin] if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Linear({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, nonlin = None, **kwargs):
        self.neurons = [Neuron(nin, nonlin, **kwargs) for _ in range(nout)]
        self.nin = nin
        self.nout = nout

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer : {self.nin} -> {self.nout}\n"

class Net(Module):

    def __init__(self, arch):
        self.layers = arch
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"{''.join(str(layer) for layer in self.layers)}"


