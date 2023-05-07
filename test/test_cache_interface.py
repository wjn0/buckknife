import unittest

import os

from time import sleep

import ray

import torch
from torch.utils.data import Dataset

from buckknife.caching import cache


class CacheDecoratorTest(unittest.TestCase):
    def setUpClass():
        ray.init(num_cpus=1, num_gpus=0)

    def setUp(self):
        self.cached_f = cache()(f)
        self.f_cache = f.__cache__

    def test_function_works(self):
        assert self.cached_f(2) == 4

    def test_function_caches(self):
        assert self.cached_f(2) == 4
        assert ray.get(self.f_cache.num_items.remote()) == 1

    def tearDownClass():
        ray.shutdown()


def f(x):
    return x + 2
