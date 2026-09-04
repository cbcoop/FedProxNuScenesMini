import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedprox_nuscenes.config import ExperimentConfig
from fedprox_nuscenes.data import (
    FourModalEmbeddingDataset,
    load_embedding_cache,
    partition_scenes,
    split_by_scene,
)
from fedprox_nuscenes.evaluation import classification_metrics, collect_predictions
from fedprox_nuscenes.federated import client_schedule, train_federated
from fedprox_nuscenes.model import PresenceAwareFusion
from fedprox_nuscenes.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/final_run"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        cache_path=args.cache,
        results_dir=args.output,
        seed=args.seed,
        mu=args.mu,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_embedding_cache(cfg.cache_path)
    split = split_by_scene(cache["scene_ids"], seed=cfg.split_seed)
    datasets = {
        "train": FourModalEmbeddingDataset(cache, split.train),
        "validation": FourModalEmbeddingDataset(cache, split.validation),
        "test": FourModalEmbeddingDataset(cache, split.test),
    }
    client_indices, partition = partition_scenes(
        split.train, cache["scene_ids"], cache["y"], cfg.num_clients,
        cfg.num_classes, seed=cfg.seed,
    )
    clients = [FourModalEmbeddingDataset(cache, rows) for rows in client_indices]
    validation_loader = DataLoader(
        datasets["validation"], batch_size=cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False)

    class_counts = torch.bincount(datasets["train"].labels, minlength=cfg.num_classes)
    class_weights = class_counts.sum() / (cfg.num_classes * class_counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Set the seed before making the model so repeated runs start the same way.
    set_seed(cfg.seed)
    model = PresenceAwareFusion(
        cfg.embedding_dim, cfg.adapter_dim, cfg.num_classes, cfg.dropout
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())

    def validate(current_model):
        probabilities, labels, _ = collect_predictions(
            current_model, validation_loader, device
        )
        metrics = classification_metrics(probabilities, labels)
        return metrics["accuracy"], metrics["macro_f1"]

    schedule = client_schedule(
        cfg.num_clients, cfg.communication_rounds, cfg.participation_rate, cfg.seed
    )
    model.load_state_dict(initial_state)
    model, best_f1, history = train_federated(
        model,
        clients,
        validate,
        criterion,
        device,
        seed=cfg.seed,
        mu=cfg.mu,
        rounds=cfg.communication_rounds,
        local_epochs=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        participation_rate=cfg.participation_rate,
        patience=cfg.early_stopping_patience,
        schedule=schedule,
    )

    probabilities, labels, gates = collect_predictions(model, test_loader, device)
    metrics = classification_metrics(probabilities, labels)
    saved_metrics = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }
    saved_metrics["best_validation_macro_f1"] = best_f1
    saved_metrics["mean_gate_weights"] = gates.mean(axis=0).tolist()
    (cfg.results_dir / "metrics.json").write_text(json.dumps(saved_metrics, indent=2))
    pd.DataFrame(history).to_csv(cfg.results_dir / "training_history.csv", index=False)
    pd.DataFrame(partition).to_json(
        cfg.results_dir / "client_partition.json", orient="records", indent=2
    )
    torch.save(
        {"model_state": model.state_dict(), "config": cfg.__dict__, "seed": cfg.seed},
        cfg.results_dir / "model.pt",
    )
    print(json.dumps(saved_metrics, indent=2))


if __name__ == "__main__":
    main()
