import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        for layer in model:
            x = layer(x)

            if isinstance(layer, nn.ReLU):
                dead_count = 0

                for i in range(x.shape[1]):
                    if (x[:, i] == 0).all():
                        dead_count += 1

                dead_fractions.append(dead_count / x.shape[1])

        return [round(v, 4) for v in dead_fractions]

    def suggest_fix(self, dead_fractions: List[float]) -> str:

        match True:

            case _ if any(f > 0.5 for f in dead_fractions):
                return 'use_leaky_relu'

            case _ if dead_fractions[0] > 0.3:
                return 'reinitialize'

            case _ if (
                dead_fractions[-1] > 0.1
                and all(dead_fractions[i] < dead_fractions[i + 1]
                        for i in range(len(dead_fractions) - 1))
            ):
                return 'reduce_learning_rate'

            case _ if max(dead_fractions) < 0.1:
                return 'healthy'

            case _:
                return 'healthy'