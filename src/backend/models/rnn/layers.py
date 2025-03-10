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

