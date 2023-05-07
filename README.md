# buckknife

A collection of data science and machine learning utilities that I've factored out of past and ongoing Python projects for more easily-maintainable reuse.

## Directory

* [`RayDataLoader`](buckknife/pytorch/data/ray_dataloader.py): A [`DataLoader`](https://pytorch.org/docs/stable/data.html) which farms out each batch fetch to a [Ray](https://www.ray.io/) cluster as a task (if one is available).
