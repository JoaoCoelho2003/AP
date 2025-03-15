from abc import abstractmethod
import numpy as np

class LossFunction:
    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true

class BinaryCrossEntropy(LossFunction):
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (p - y_true) / (p * (1 - p) + self.epsilon)

class FocalLoss(LossFunction):
    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1e-15):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        ce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return np.mean(focal_weight * ce)
    
    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        bce_grad = (y_pred - y_true) / (y_pred * (1 - y_pred) + self.epsilon)
        focal_grad = self.gamma * p_t * np.log(p_t + self.epsilon) * focal_weight
        
        return focal_weight * bce_grad + focal_grad

