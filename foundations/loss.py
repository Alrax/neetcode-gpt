import numpy as np
from numpy.typing import NDArray


class Solution:

    e = 1e-7

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        loss = -1/len(y_true) * sum(y_true * np.log(y_pred + self.e) + (1 - y_true) * np.log(1 - y_pred + self.e))
        
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        loss = -1/len(y_true) * sum(sum(y_true*np.log(y_pred + self.e)))

        return round(loss, 4)
