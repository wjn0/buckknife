from abc import ABCMeta

from numbers import Number

import numpy as np

import pandas as pd

import pyhash

import ray



# A fairly efficient hash function, but can of course be overwritten.
hasher = pyhash.xx_64()


@ray.remote
class CacheActor:
    """
    Cache objects in the Ray object store.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._cache = {}

    def set(self, key, value):
        """
        Set a value in the store.

        Params:
            key: A hashable Python object that identifies the value.
            value: The Pickleable Python object to be cached.

        Returns:
            success: A boolean indicating whether the cache operation was successful.
        """
        key_hash = CacheActor.hash_python_object(key)
        self._cache[key_hash] = value
        success = True

        return success
    
    def get(self, key):
        """
        Get a value from the store.

        Params:
            key: A hashable Python object that identifies the value.

        Returns:
            obj: The object in the store.

        Raises:
            KeyError: If the object could not be found.
        """
        key_hash = CacheActor.hash_python_object(key)
        if key_hash in self._cache:
            obj = self._cache[key_hash]
            return obj

        raise KeyError

    def contains(self, key):
        """
        Check whether a key is available in the store.

        Params:
            key: A hashable Python object to check.

        Returns:
            present: Whether the key is present in the store.
        """
        key_hash = CacheActor.hash_python_object(key)
        return key_hash in self._cache

    def set_by_hash(self, key_hash, value):
        """
        Set a value in the store using an already-hashed key.

        Params:
            key_hash: The hash of a Python object.
            value: The `pickle`able Python object to be cached.

        Returns:
            success: A boolean indicating whether the cache operation was successful.
        """
        self._cache[key_hash] = value
        success = True
        
        return success

    def get_by_hash(self, key_hash):
        """
        Get a value from the store using an already-hashed key.

        Params:
            key_hash: The hash of a Python object.

        Returns:
            obj: The object in the store.

        Raises:
            KeyError: If the object could not be found.
        """
        if key_hash in self._cache:
            obj = self._cache[key_hash]
            return obj

        raise KeyError

    def contains_by_hash(self, key_hash):
        """
        Check whether a key is available in the store by its hash.

        Params:
            key_hash: The key's hash.

        Returns:
            present: Whether the key is present in the store.
        """
        return key_hash in self._cache

    def num_items(self):
        """Check the number of items in the cache."""
        return len(self._cache)

    @staticmethod
    def hash_python_object(arg):
        return _hash_python_object(arg)


def _hash_python_object(arg):
    if isinstance(arg, dict):
        return _hash_dict(arg)
    elif isinstance(arg, Number):
        return _hash_number(arg)
    elif isinstance(arg, np.ndarray):
        return _hash_np(arg)
    elif isinstance(arg, bool):
        return _hash_bool(arg)
    elif isinstance(arg, str):
        return _hash_str(arg)
    elif isinstance(arg, ABCMeta):
        return _hash_class(arg)
    elif arg is None:
        return NONE_VAL
    elif isinstance(arg, pd.DataFrame):
        return _hash_df(arg)
    elif isinstance(arg, list):
        return _hash_list(arg)
    elif isinstance(arg, tuple):
        return hash(tuple(_hash_python_object(el) for el in arg))
    elif inspect.isclass(arg):
        return _hash_class(arg)
    elif callable(arg):
        return _hash_callable(arg)
    else:
        raise ValueError(f"Not sure how to hash {arg}")


def _hash_number(arg):
    return (0, hash(arg))


def _hash_np(arg):
    return (1, hasher(arg.tobytes()))


def _hash_bool(arg):
    return (2, hash(arg))


def _hash_dict(arg):
    return (
        3,
        hash(tuple(_hash_python_object(el) for el in arg.keys())),
        hash(tuple(_hash_python_object(el) for el in arg.values())),
    )


def _hash_str(arg):
    return (4, hasher(arg))


def _hash_class(arg):
    return (5, _hash_str(str(arg)))


def _hash_df(arg):
    return (6, _hash_dict(arg.to_dict()))


def _hash_estimator(arg):
    return (7, _hash_class(type(arg)), _hash_dict(arg.get_params()))


def _hash_list(arg):
    return (8, _hash_python_object(tuple(arg)))


def _hash_callable(arg):
    return (9, _hash_python_object(arg.__name__))
