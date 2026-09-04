from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


def sample_presence(batch_size: int, device, modality_dropout: float,
                    all_missing_rate: float) -> torch.Tensor:
    mask = (torch.rand(batch_size, 4, device=device) > modality_dropout).float()
    all_missing = torch.rand(batch_size, device=device) < all_missing_rate
    mask[all_missing] = 0
    return mask


def train_local_client(
    global_model: torch.nn.Module,
    dataset: Dataset,
    criterion,
    device,
    mu: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    modality_dropout: float = 0.20,
    all_missing_rate: float = 0.02,
):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in global_model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = AdamW(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    task_total = 0.0
    proximal_total = 0.0
    seen = 0

    for _ in range(local_epochs):
        for batch in loader:
            inputs = [batch[name].to(device) for name in ("vision", "radar", "depth", "motion")]
            targets = batch["label"].to(device)
            presence = sample_presence(
                len(targets), device, modality_dropout, all_missing_rate
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = local_model(*inputs, presence_mask=presence)
            task_loss = criterion(logits, targets)
            proximal_norm = torch.zeros((), device=device)
            if mu > 0:
                proximal_norm = sum(
                    torch.sum((parameter - anchor[name]) ** 2)
                    for name, parameter in local_model.named_parameters()
                    if parameter.requires_grad
                )
            loss = task_loss + 0.5 * mu * proximal_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()
            task_total += task_loss.item() * len(targets)
            proximal_total += proximal_norm.item() * len(targets)
            seen += len(targets)

    state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in local_model.state_dict().items()
    )
    diagnostics = {
        "task_loss": task_total / max(seen, 1),
        "proximal_norm_sq": proximal_total / max(seen, 1),
    }
    return state, len(dataset), diagnostics


def weighted_average(updates):
    if not updates:
        raise ValueError("At least one client update is required")
    total = sum(n_samples for _, n_samples, _ in updates)
    if total <= 0:
        raise ValueError("Client sample counts must sum to a positive number")
    first = updates[0][0]
    averaged = OrderedDict()
    for name, tensor in first.items():
        if torch.is_floating_point(tensor):
            averaged[name] = sum(
                state[name] * (n_samples / total) for state, n_samples, _ in updates
            )
        else:
            averaged[name] = tensor.clone()
    return averaged


def client_schedule(num_clients: int, rounds: int, participation_rate: float,
                    seed: int) -> list[np.ndarray]:
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    n_selected = min(num_clients, max(2, int(np.ceil(num_clients * participation_rate))))
    schedule = []
    for round_id in range(1, rounds + 1):
        rng = np.random.default_rng(seed * 100_000 + round_id)
        schedule.append(np.sort(rng.choice(num_clients, n_selected, replace=False)))
    return schedule


def train_federated(
    model: torch.nn.Module,
    client_datasets: Sequence[Dataset],
    validation_fn: Callable[[torch.nn.Module], tuple[float, float]],
    criterion,
    device,
    *,
    seed: int,
    mu: float = 0.01,
    rounds: int = 30,
    local_epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    participation_rate: float = 1.0,
    patience: int = 10,
    schedule: Sequence[np.ndarray] | None = None,
):
    if schedule is None:
        schedule = client_schedule(len(client_datasets), rounds, participation_rate, seed)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -np.inf
    stale_rounds = 0
    history = []
    for round_id, participants in enumerate(schedule, start=1):
        updates = []
        for client_id in participants:
            updates.append(train_local_client(
                model, client_datasets[int(client_id)], criterion, device,
                mu=mu, local_epochs=local_epochs, learning_rate=learning_rate,
                weight_decay=weight_decay, batch_size=batch_size,
                seed=seed * 100_000 + round_id * 100 + int(client_id),
            ))
        model.load_state_dict(weighted_average(updates))
        model.to(device)
        val_accuracy, val_f1 = validation_fn(model)
        history.append({
            "round": round_id,
            "participants": [int(value) for value in participants],
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
            "mu": mu,
        })
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= patience:
                break
    model.load_state_dict(best_state)
    return model, float(best_f1), history
