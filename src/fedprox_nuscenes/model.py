from __future__ import annotations

import torch
from torch import nn


class ModalityAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


class PresenceAwareFusion(nn.Module):
    def __init__(self, embedding_dim: int = 1024, adapter_dim: int = 256,
                 num_classes: int = 3, dropout: float = 0.35):
        super().__init__()
        self.modalities = ("vision", "radar", "depth", "motion")
        self.depth_pre_norm = nn.LayerNorm(embedding_dim)
        self.adapters = nn.ModuleDict({
            name: ModalityAdapter(embedding_dim, adapter_dim) for name in self.modalities
        })
        self.gate = nn.Sequential(
            nn.Linear(4 * adapter_dim + 4, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, 4),
        )
        self.no_input_embedding = nn.Parameter(torch.zeros(adapter_dim))
        self.classifier = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, num_classes),
        )

    def forward(self, vision, radar, depth, motion, presence_mask=None):
        inputs = {"vision": vision, "radar": radar, "depth": depth, "motion": motion}
        batch_size = vision.size(0)
        if presence_mask is None:
            presence_mask = vision.new_ones((batch_size, 4))
        inputs["depth"] = self.depth_pre_norm(inputs["depth"])
        adapted = torch.stack(
            [self.adapters[name](inputs[name]) for name in self.modalities], dim=1
        )
        adapted = adapted * presence_mask.unsqueeze(-1)
        gate_input = torch.cat((adapted.flatten(start_dim=1), presence_mask), dim=1)
        gate_logits = self.gate(gate_input).masked_fill(presence_mask == 0, -1e9)
        gate_weights = torch.softmax(gate_logits, dim=1) * presence_mask
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused = (adapted * gate_weights.unsqueeze(-1)).sum(dim=1)
        no_input = presence_mask.sum(dim=1, keepdim=True) == 0
        fallback = self.no_input_embedding.unsqueeze(0).expand(batch_size, -1)
        fused = torch.where(no_input, fallback, fused)
        return self.classifier(fused), gate_weights
