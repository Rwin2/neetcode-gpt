import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        #forward
        z1=np.dot(W1,x)+b1
        a1=np.maximum(z1,0)
        y_hat=np.dot(W2,a1)+b2
        loss=np.mean( (y_hat-y_true)**2)

        # Backward
        n   = len(y_true)
        dz2 = (2 / n) * (y_hat - y_true)

        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0)           # masque binaire ReLU

        dW1 = np.outer(dz1, x)
        db1 = dz1

        r=lambda x:np.round(x+0.0,4).tolist()
        return {"loss":round(float(loss),4),
                "dW1":r(dW1),
                "db1":r(db1),
                "dW2":r(dW2),
                "db2":r(db2)}

