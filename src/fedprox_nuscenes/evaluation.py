from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(ece)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    target = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def collect_predictions(model, loader, device, presence_matrix=None):
    model.eval()
    probabilities, labels, gates = [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["label"])
            inputs = [batch[name].to(device) for name in ("vision", "radar", "depth", "motion")]
            presence = None
            if presence_matrix is not None:
                presence = torch.as_tensor(
                    presence_matrix[offset:offset + batch_size], device=device
                )
            logits, gate = model(*inputs, presence_mask=presence)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
            gates.append(gate.cpu().numpy())
            offset += batch_size
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(gates)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "ece": expected_calibration_error(probabilities, labels),
        "brier": multiclass_brier(probabilities, labels),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }
