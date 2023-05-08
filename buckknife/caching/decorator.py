"""Cache a function call."""

import ray

from .actor import CacheActor


def cache():
    """
    Cache a function using the Ray object store.

    Example:
        @cache()
        def f(x):
            return x + 2
    """
    return _cache


def _cache(fxn):
    f_name = fxn.__name__
    cache_name = f"cache_{f_name}"
    fxn.__cache__: CacheActor = CacheActor.remote(name=cache_name)

    def f_star(*args, **kwargs):
        key = {
            "type": "cache_function",
            "function_name": f_name,
            "args": args,
            "kwargs": kwargs,
        }
        key_hash = CacheActor.hash_python_object(key)
        if ray.get(fxn.__cache__.contains_by_hash.remote(key_hash)):
            result = ray.get(fxn.__cache__.get_by_hash.remote(key_hash))
        else:
            result = fxn(*args, **kwargs)
            fxn.__cache__.set_by_hash.remote(key_hash, result)

        return result

    return f_star
