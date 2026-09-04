import numpy as np

from fedprox_nuscenes.missingness import make_presence_mask, mask_pattern_counts


def test_mcar_mask_has_requested_marginal_rate():
    mask = make_presence_mask(100, [0, 1, 2, 3], 0.30, np.random.default_rng(7))
    assert np.all((mask == 0).sum(axis=0) == 30)
    assert sum(mask_pattern_counts(mask).values()) == 100
