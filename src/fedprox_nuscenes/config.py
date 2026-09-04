from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    cache_path: Path = Path("data/processed/imagebind_nuscenes_embeddings.npz")
    results_dir: Path = Path("results")
    seed: int = 42
    split_seed: int = 264
    batch_size: int = 32

    embedding_dim: int = 1024
    adapter_dim: int = 256
    dropout: float = 0.35
    num_classes: int = 3
    modalities: tuple[str, ...] = ("vision", "radar", "depth", "motion")

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    communication_rounds: int = 30
    local_epochs: int = 1
    early_stopping_patience: int = 10
    mu: float = 0.01
    num_clients: int = 4
    participation_rate: float = 1.0

    modality_dropout_rate: float = 0.20
    all_missing_rate: float = 0.02
    class_names: tuple[str, ...] = field(
        default=("vehicle", "pedestrian", "static_obstacle")
    )
