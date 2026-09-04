from __future__ import annotations

import numpy as np


def class_centroid_distance(embeddings: np.ndarray, labels: np.ndarray,
                            train_indices: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(labels), dtype=np.float32)
    for label in np.unique(labels):
        train_rows = train_indices[labels[train_indices] == label]
        centroid = embeddings[train_rows].mean(axis=0)
        class_rows = np.flatnonzero(labels == label)
        scores[class_rows] = np.linalg.norm(embeddings[class_rows] - centroid, axis=1)
    low, high = scores.min(), scores.max()
    return np.zeros_like(scores) if high == low else (scores - low) / (high - low)


def make_presence_mask(n_samples: int, target_modalities: list[int], rate: float,
                       rng: np.random.Generator, scores: np.ndarray | None = None):
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    mask = np.ones((n_samples, 4), dtype=np.float32)
    number_missing = int(round(n_samples * rate))
    for modality in target_modalities:
        probabilities = None
        if scores is not None:
            probabilities = np.exp(3.0 * scores[:, modality])
            probabilities /= probabilities.sum()
        rows = rng.choice(n_samples, number_missing, replace=False, p=probabilities)
        mask[rows, modality] = 0
    return mask


def mask_pattern_counts(mask: np.ndarray) -> dict[str, int]:
    patterns = ["".join(str(int(value)) for value in row) for row in mask]
    values, counts = np.unique(patterns, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))
