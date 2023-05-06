# A Ray-based dataloader for PyTorch



## Example

```python
from time import sleep

import torch
from torch.utils.data import Dataset

from ray_pytorch_dataloader import RayDataLoader

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1_000

    def __getitem__(self, idx):
        # Do lots of CPU-bound work...
        sleep(60)

        return torch.randn(3, 1000, 1000)


def train():
    dataset = MyDataset()
    dataloader = RayDataLoader(dataset, batch_size=2, remote_args={"num_cpus": 1})
    for batch in dataloader:
        take_step(batch)

import ray
ray.init()  # without this, will run transparently as if using a regular `DataLoader`

train()
```

## How it works

I always forget how Ray works, but often want to use it in this context. Roughly:

* Ray allows Python processes on different nodes to talk to each other when they're a member
  of the same Ray cluster.
* Start a Ray cluster with `ray start --head --num-cpus=X`. Connect to it with `ray start --address=hostname:port --num-cpus=X`.

