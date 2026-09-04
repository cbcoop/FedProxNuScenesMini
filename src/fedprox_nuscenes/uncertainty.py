from __future__ import annotations

import numpy as np
import torch
from torch import nn


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def decompose_mc_probabilities(pass_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    # pass_probabilities has shape: passes x samples x classes
    mean_probability = pass_probabilities.mean(axis=0)
    total = entropy(mean_probability)
    expected_entropy = entropy(pass_probabilities).mean(axis=0)
    return {
        "mean_probability": mean_probability,
        "total_uncertainty": total,
        "aleatoric_proxy": expected_entropy,
        "epistemic_uncertainty": total - expected_entropy,
    }
