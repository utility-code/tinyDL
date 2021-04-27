from abc import abstractmethod, ABCMeta
import collections

from .tensor import Tensor
from .layers import Linear, Node


class Model(metaclass=ABCMeta):
    """

    Args:
        metaclass . Defaults to ABCMeta.
    Creates a model with GPU support and backwards/forward prop
    """

    def __init__(self, *args, **kargs):
        self._total_params = 0

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _modules(self):
        _modules = []
        for _, layer in self.__dict__.items():
            if isinstance(layer, Linear):
                _modules.append(layer)
        return _modules

    def init_backward(self):
        for p in self.parameters():
            p.grad = 0.0
        return

    @property
    def modules(self):
        return self._modules()

    @property
    def total_params(self):
        return self._total_params

    def parameters(self):
        _modules = self._modules()
        params = [p for hidden in _modules for p in hidden.parameters()]
        self._total_params = len(params)
        return params

    def summary(self):
        """
        Pretty print model anyone?
        """
        _modules = self.modules
        _format_summary = "Model summary\n------"
        for key, mod in enumerate(_modules):
            summary = mod.summary()
            name = f"{key} {summary['name']}"
            in_shape = summary["shape"][0]
            out_shape = summary["shape"][1]
            params = summary["params"]
            _format_summary += f"\n{name}: {in_shape} -> {out_shape}"
        print(_format_summary + "\n------\n\n")
