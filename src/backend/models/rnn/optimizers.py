import numpy as np

class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return param - self.learning_rate * grad

class AdamOptimizer:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param