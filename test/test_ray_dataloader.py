import unittest

import os

from time import sleep

import ray

import torch
from torch.utils.data import Dataset

from buckknife.pytorch.data import RayDataLoader


class RayDataLoaderTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=1, num_gpus=0)

        self.dataset = MyDataset()

    def test_gets_batch(self):
        dataloader = RayDataLoader(self.dataset, batch_size=2, prefetch_factor=2)
        batch = next(iter(dataloader))

        assert batch.size(0) == 2

    def tearDown(self):
        ray.shutdown()


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1_000

    def __getitem__(self, idx):
        return torch.randn(3, 1000, 1000)
