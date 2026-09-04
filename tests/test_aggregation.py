from collections import OrderedDict

import torch

from fedprox_nuscenes.federated import weighted_average


def test_weighted_average_uses_client_sample_counts():
    first = OrderedDict(weight=torch.tensor([1.0, 3.0]))
    second = OrderedDict(weight=torch.tensor([5.0, 7.0]))
    result = weighted_average([(first, 1, {}), (second, 3, {})])
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))


def test_weighted_average_keeps_integer_buffers():
    first = OrderedDict(counter=torch.tensor(2, dtype=torch.long))
    second = OrderedDict(counter=torch.tensor(8, dtype=torch.long))
    result = weighted_average([(first, 1, {}), (second, 3, {})])
    assert result["counter"].item() == 2
