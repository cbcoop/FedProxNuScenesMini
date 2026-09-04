# Limitations

The project uses nuScenes-mini, which only has ten scenes. The client sizes and
class distributions are also uneven. The current model depends heavily on
vision and makes many errors on static obstacles.

Some early notebook comparisons did not use the same starting model and missing
data masks for every setting. The organized training code fixes the model seed,
but the final FedAvg and FedProx comparison still needs to be rerun under the
same conditions.
