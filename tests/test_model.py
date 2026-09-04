import torch

from fedprox_nuscenes.model import PresenceAwareFusion


def test_model_shapes_and_masked_gate():
    model = PresenceAwareFusion(embedding_dim=8, adapter_dim=4, num_classes=3)
    inputs = [torch.randn(5, 8) for _ in range(4)]
    mask = torch.ones(5, 4)
    mask[:, 1] = 0
    logits, gates = model(*inputs, presence_mask=mask)
    assert logits.shape == (5, 3)
    assert gates.shape == (5, 4)
    assert torch.all(gates[:, 1] == 0)
    assert torch.allclose(gates.sum(dim=1), torch.ones(5))


def test_all_missing_uses_fallback_embedding():
    model = PresenceAwareFusion(embedding_dim=8, adapter_dim=4, num_classes=3)
    inputs = [torch.randn(2, 8) for _ in range(4)]
    logits, gates = model(*inputs, presence_mask=torch.zeros(2, 4))
    assert torch.isfinite(logits).all()
    assert torch.all(gates == 0)
