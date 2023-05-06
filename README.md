# buckknife

A collection of utilities that I've factored out of ongoing projects for more easily-maintainable reuse.

## Directory

* [`RayDataLoader`](buckknife/pytorch/data/ray_dataloader.py): A [`DataLoader`](https://pytorch.org/docs/stable/data.html) which farms out each batch fetch to a Ray cluster as a task (if one is available).
