import numpy as np
from typing import Optional


class Perceptron:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr: float = learning_rate
        self.n_iterations: int = n_iterations
        self.bias: Optional[float] = None
        self.bias: Optional[float] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(x):
                linear_output: float = np.dot(x_i, self.weights) + self.bias

                if y[idx] * linear_output <= 0:
                    self.weights += self.lr * y[idx] * x_i
                    self.bias += self.lr * y[idx]
