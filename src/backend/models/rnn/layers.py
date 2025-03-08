from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.input_shape = None

    @abstractmethod
    def initialize(self, optimizer):
        pass

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def output_shape(self) -> tuple:
        pass

    @abstractmethod
    def parameters(self) -> int:
        pass

class DropoutLayer(Layer):
    def __init__(self, rate: float):
        assert 0.0 <= rate < 1.0, "Dropout rate must be between 0 and 1."
        self.rate = rate
        self.mask = None

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) / (1 - self.rate)
            return input * self.mask
        return input

    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad * self.mask if self.mask is not None else accum_grad

    def output_shape(self) -> tuple:
        return self.input_shape

    def parameters(self) -> int:
        return 0
