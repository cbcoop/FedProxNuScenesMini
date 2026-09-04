FedProx nuScenes-mini project

This repository contains the code and results from my nuScenes-mini project.
The task is object classification using camera, radar, depth, and motion
features. The classes are vehicle, pedestrian, and static obstacle.

The four training scenes are treated as four clients. The project compares
FedAvg and FedProx and also tests what happens when sensor inputs are missing.

The original work is in `notebooks/FedProxyStart.ipynb`. The reusable code is
in `src/fedprox_nuscenes`. Saved plots and smaller result files are in
`results`.

To install the packages:

'pip install -r requirements.txt'
'pip install -e'


The nuScenes data, embedding cache, and trained model files are not included
because they are too large for this repository.
