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

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_output: np.ndarray = np.dot(x, self.weights) + self.bias
        return np.sign(linear_output)


if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 5]])
    y = np.array([1, 1, -1, -1])

    model = Perceptron(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
