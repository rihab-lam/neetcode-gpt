import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        # Convert lists to NumPy arrays
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # Batch statistics for each feature
            mean_b = np.mean(x, axis=0)
            var_b = np.var(x, axis=0)

            # Normalize
            x_hat = (x - mean_b) / np.sqrt(var_b + eps)

            # Affine transformation
            y = gamma * x_hat + beta

            # Update running statistics
            running_mean = (
                (1 - momentum) * running_mean
                + momentum * mean_b
            )

            running_var = (
                (1 - momentum) * running_var
                + momentum * var_b
            )

        else:
            # Normalize using running statistics
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

            # Affine transformation
            y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )