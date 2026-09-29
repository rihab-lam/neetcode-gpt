import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        output=[]
        x_hat=[]
        for i in range(len(x)):
            rms = np.sqrt(np.mean(np.array(x)**2) + eps)
            x_hat.append(x[i]/rms)
        for i in range (len(gamma)):
            output.append(np.round(gamma[i]*x_hat[i],4))
        return output
