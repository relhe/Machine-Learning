import numpy as np
from typing import Optional


class Perceptron:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to ensure they are -1 or 1
        y = np.where(y <= 0, -1, 1).flatten()

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y[idx] * linear_output <= 0:
                    self.weights += self.lr * y[idx] * x_i
                    self.bias += self.lr * y[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)


if __name__ == "__main__":
    # Generate synthetic dataset
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = np.array([1 if i > 0 else -1 for i in X]).reshape(-1)

    X_test = np.linspace(-20, 20, 100).reshape(-1, 1)
    y_test = np.array([1 if i > 0 else -1 for i in X_test]).reshape(-1)

    # Train custom perceptron classifier
    model = Perceptron(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X_test)

    # Evaluate custom perceptron classifier
    accuracy_custom = np.mean(predictions == y_test)
    print(f"Custom Perceptron classifier Accuracy: {accuracy_custom}")

    # Scikit-learn Perceptron for comparison
    from sklearn.linear_model import Perceptron
    sklearn_model = Perceptron(
        eta0=0.01, max_iter=1000, random_state=42, tol=1e-3)
    sklearn_model.fit(X, y)
    accuracy_sklearn = sklearn_model.score(X_test, y_test)
    print(f"Scikit-learn Perceptron Accuracy: {accuracy_sklearn}")
