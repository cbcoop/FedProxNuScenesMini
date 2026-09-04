from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SceneSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_embedding_cache(path: str | Path):
    return np.load(Path(path), allow_pickle=False)


def split_by_scene(scene_ids: np.ndarray, seed: int = 264, n_validation: int = 3,
                   n_test: int = 3) -> SceneSplit:
    # Keep complete scenes together so nearby samples do not cross splits.
    scenes = np.unique(scene_ids).copy()
    if n_validation + n_test >= len(scenes):
        raise ValueError("The split must leave at least one training scene")
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    test_scenes = scenes[:n_test]
    validation_scenes = scenes[n_test:n_test + n_validation]
    train_scenes = scenes[n_test + n_validation:]
    return SceneSplit(
        train=np.flatnonzero(np.isin(scene_ids, train_scenes)),
        validation=np.flatnonzero(np.isin(scene_ids, validation_scenes)),
        test=np.flatnonzero(np.isin(scene_ids, test_scenes)),
    )


def partition_scenes(
    train_indices: np.ndarray,
    scene_ids: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[dict]]:
    # A scene stays with one client. Start with the largest scenes.
    scenes = np.unique(scene_ids[train_indices]).copy()
    if not 2 <= num_clients <= len(scenes):
        raise ValueError("num_clients must be between 2 and the number of training scenes")
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    records = []
    for scene in scenes:
        idx = train_indices[scene_ids[train_indices] == scene]
        counts = np.bincount(labels[idx], minlength=num_classes)
        records.append((scene, idx, counts))
    records.sort(key=lambda item: len(item[1]), reverse=True)

    assignments: list[list[tuple[str, np.ndarray]]] = [[] for _ in range(num_clients)]
    sizes = np.zeros(num_clients, dtype=int)
    class_counts = np.zeros((num_clients, num_classes), dtype=int)
    for scene, idx, counts in records:
        scores = []
        for client_id in range(num_clients):
            projected = class_counts[client_id] + counts
            scores.append((sizes[client_id], projected.max() - projected.min(), client_id))
        client_id = min(scores)[2]
        assignments[client_id].append((scene, idx))
        sizes[client_id] += len(idx)
        class_counts[client_id] += counts

    client_indices = [
        np.concatenate([idx for _, idx in assigned]).astype(int)
        for assigned in assignments
    ]
    summary = [
        {
            "client": client_id,
            "n_objects": int(sizes[client_id]),
            "scenes": [str(scene) for scene, _ in assignments[client_id]],
            "class_counts": class_counts[client_id].tolist(),
        }
        for client_id in range(num_clients)
    ]
    return client_indices, summary


class FourModalEmbeddingDataset(Dataset):
    def __init__(self, cache, indices: np.ndarray):
        self.vision = torch.from_numpy(cache["vision"][indices]).float()
        self.radar = torch.from_numpy(cache["radar"][indices]).float()
        self.depth = torch.from_numpy(cache["depth"][indices]).float()
        # Older caches use the name imu for the ego-motion embedding.
        motion_key = "motion" if "motion" in cache.files else "imu"
        self.motion = torch.from_numpy(cache[motion_key][indices]).float()
        self.labels = torch.from_numpy(cache["y"][indices]).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "vision": self.vision[index],
            "radar": self.radar[index],
            "depth": self.depth[index],
            "motion": self.motion[index],
            "label": self.labels[index],
        }
