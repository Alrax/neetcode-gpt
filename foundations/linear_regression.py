import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        Y = []
        for i in range(len(X)):
            y_pred = sum(X[i] * weights)
            Y.append(y_pred)

        return np.round(Y, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        
        MSE = np.sum((model_prediction - ground_truth) ** 2) / len(ground_truth)
        return round(MSE, 5)
