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


def _cache(f):
    f_name = f.__name__
    cache_name = f"cache_{f_name}"
    f.__cache__: CacheActor = CacheActor.remote(name=cache_name)

    def f_star(*args, **kwargs):
        key = {"type": "cache_function",
               "function_name": f_name,
               "args": args,
               "kwargs": kwargs}
        key_hash = CacheActor.compute_hash(key)
        if f.__cache__.contains_key_hash(key_hash):
            result = ray.get(f.__cache__.get_by_hash.remote(key_hash))
        else:
            result = f(*args, **kwargs)
            f.__cache__.set_by_hash.remote(key_hash, result)

        return result

    return f_star
