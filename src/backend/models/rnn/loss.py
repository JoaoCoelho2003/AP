import numpy as np

class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        grad = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return grad

