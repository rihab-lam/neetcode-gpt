import numpy as np
from typing import List


class Solution:

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert to NumPy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # -----------------
        # Forward pass
        # -----------------

        # x -> Linear
        z1 = np.dot(W1, x) + b1

        # ReLU
        a1 = np.maximum(0, z1)

        # Linear
        # W2 has shape (output_size, hidden_size)
        z2 = np.dot(W2, a1) + b2

        # MSE
        loss = np.mean((z2 - y_true) ** 2)

        # -----------------
        # Backward pass
        # -----------------

        # dL/dz2
        dz2 = 2 * (z2 - y_true) / len(y_true)

        # dL/dW2
        dW2 = np.outer(dz2, a1)

        # dL/db2
        db2 = dz2

        # dL/da1
        da1 = np.dot(W2.T, dz2)

        # ReLU derivative
        dz1 = da1 * (z1 > 0)

        # dL/dW1
        dW1 = np.outer(dz1 , x)

        # dL/db1
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }