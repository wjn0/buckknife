import unittest

import os

from time import sleep

import ray

import torch
from torch.utils.data import Dataset

from buckknife.caching.actor import CacheActor


class CacheActorTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=1, num_gpus=0)

        self.actor = CacheActor.remote(name="test_cache")

    def test_sets_key(self):
        self.actor.set.remote("test_key", "test_value")

        assert ray.get(self.actor.contains.remote("test_key"))

    def test_value_gettable(self):
        self.actor.set.remote("test_key1", "my_value")

        assert ray.get(self.actor.get.remote("test_key1")) == "my_value"

    def test_key_contains_by_hash(self):
        self.actor.set.remote("test_key2", "my_value2")
        hash_key = CacheActor.hash_python_object("test_key2")

        assert ray.get(self.actor.contains_by_hash.remote(hash_key))
        assert ray.get(self.actor.get_by_hash.remote(hash_key)) == "my_value2"

    def test_set_by_hash(self):
        hash_key = CacheActor.hash_python_object("test_key3")
        self.actor.set_by_hash.remote(hash_key, "my_value3")

        assert ray.get(self.actor.contains.remote("test_key3"))
        assert ray.get(self.actor.get.remote("test_key3")) == "my_value3"

    def tearDown(self):
        ray.shutdown()
