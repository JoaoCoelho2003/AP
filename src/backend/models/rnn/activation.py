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

class SigmoidActivation(ActivationLayer):
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.activation_function(x)
        return s * (1 - s)

class ReLUActivationRnn(ActivationLayer):
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
