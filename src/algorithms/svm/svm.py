import numpy as np
from typing import Optional
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000) -> None:
        """
        Support Vector Machine implementation from scratch.
        Parameters:
        - learning_rate: Learning rate for gradient descent.
        - lambda_param: Regularization parameter.
        - n_iters: Number of iterations for training.
        """
        self.lr: float = learning_rate
        self.lambda_param: float = lambda_param
        self.n_iters: int = n_iters
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM model to the training data.
        Parameters:
        - X: Input data of shape (n_samples, n_features).
        - y: Target labels of shape (n_samples,).
             Labels should be -1 or 1.
        """
        _, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Ensure labels are -1 and 1
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No margin violation
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Margin violation
                    self.w -= self.lr * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        Parameters:
        - X: Input data of shape (n_samples, n_features).
        Returns:
        - Predicted labels of shape (n_samples,).
        """
        if self.w is None or self.b is None:
            raise ValueError(
                "The model has not been trained yet. Call `fit` before `predict`.")

        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == "__main__":

    # Generate toy dataset
    X, y = make_blobs(n_samples=100, centers=2,
                      random_state=42, cluster_std=1.0)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    # Train the custom SVM
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)

    # Predict using the custom SVM
    predictions = svm.predict(X)

    # Visualize the decision boundary
    def plot_decision_boundary(X, y, model, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title(title)
        plt.show()

    plot_decision_boundary(X, y, svm, "Custom SVM Decision Boundary")

    # Compare with sklearn's SVM
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    plot_decision_boundary(X, y, clf, "Scikit-learn's SVM Decision Boundary")

    print("Training accuracy of custom SVM: {:.2f}%".format(
        (predictions == y).mean() * 100))
    print("Training accuracy of sklearn's SVM: {:.2f}%".format(
        (clf.predict(X) == y).mean() * 100))
