import numpy as np

class ActivationLayer:
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Activation function not implemented.")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Derivative function not implemented.")

class TanhActivation(ActivationLayer):
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
