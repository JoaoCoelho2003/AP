import numpy as np

class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return param - self.learning_rate * grad
