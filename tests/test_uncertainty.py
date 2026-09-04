import numpy as np

from fedprox_nuscenes.uncertainty import decompose_mc_probabilities


def test_epistemic_is_zero_when_passes_agree():
    probabilities = np.array([[[0.8, 0.2], [0.4, 0.6]]] * 5)
    result = decompose_mc_probabilities(probabilities)
    assert np.allclose(result["epistemic_uncertainty"], 0.0)
