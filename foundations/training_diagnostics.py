import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        results = []

        with torch.no_grad():

            for layer in model:

                # Forward pass à travers la couche
                x = layer(x)

                # On calcule les statistiques uniquement pour Linear
                if isinstance(layer, nn.Linear):

                    mean = torch.mean(x).item()
                    std = torch.std(x).item()

                    # Un neurone est mort si sa sortie est <= 0
                    # pour tous les échantillons
                    dead_neurons = torch.all(x <= 0, dim=0)

                    dead_fraction = (
                        torch.sum(dead_neurons).item()
                        / x.shape[1]
                    )

                    results.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })

        return results
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        # Forward pass
        y_hat = model(x)

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)

        # Backward pass
        loss.backward()

        results = []

        # For each Linear layer
        for layer in model:
            if isinstance(layer, nn.Linear):

                # Get weight gradient
                grad = layer.weight.grad

                # Compute statistics
                mean = torch.mean(grad).item()
                std = torch.std(grad).item()
                norm = torch.norm(grad).item()

                # Store results
                results.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })

        return results

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # 1. Vérifier les neurones morts
        for stats in activation_stats:
            if stats["dead_fraction"] >= 0.5:
                return "dead_neurons"

        # 2. Vérifier les gradients explosifs
        for stats in gradient_stats:
            if stats["norm"] >= 10:
                return "exploding_gradients"

        # 3. Vérifier les gradients qui disparaissent
        for stats in gradient_stats:
            if stats["norm"] <= 0.001:
                return "vanishing_gradients"

        # 4. Aucun problème détecté
        return "healthy"
        

