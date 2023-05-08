"""A decorator that forces a function to finish in the required time."""

import threading

import queue

from .exceptions import Timeout


def _run_with_timeout(f, f_args, f_kwargs, timeout):
    def g(queue, *args, **kwargs):
        item = f(*args, **kwargs)
        queue.put(item)

    q = queue.Queue()
    p = threading.Thread(target=g, args=(q, *f_args), kwargs=f_kwargs)
    p.start()
    p.join(timeout)

    try:
        return q.get(False)
    except queue.Empty:
        raise Timeout


def timeout(time_limit):
    """
    Decorate a function so that it runs in the required time.

    Params:
        time_limit: The time limit in seconds.

    Returns:
        ret: The return value of the underlying function.

    Raises:
        Timeout: If the timeout limit is reached.

    Example:
        import numpy as np
        import time

        @timeout(5)
        def f(x):
            time.sleep(np.random.randn() + 5)
            return x + 2
    """
    def decorator(f):
        def f_star(*args, **kwargs):
            return _run_with_timeout(f, args, kwargs, time_limit)

        return f_star

    return decorator
