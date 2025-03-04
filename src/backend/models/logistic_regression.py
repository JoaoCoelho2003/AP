import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=300, reg_lambda=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.reg_lambda * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)


    def save(self, path="models/logistic_weights.npz"):
        np.savez(path, weights=self.weights, bias=self.bias)

    def load(self, path="models/logistic_weights.npz"):
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]