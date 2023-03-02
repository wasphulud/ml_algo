"""Decorators for the decision tree module"""

import functools
import time
import logging


def timer(func):
    """Compute and print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(
            "Finished running %r function in %.4f secs", func.__name__, run_time
        )
        return value

    return wrapper_timer


def debug(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.info("Calling %s  %s", func.__name__, signature)
        value = func(*args, **kwargs)
        logging.info("%r returned %r", func.__name__, value)
        return value

    return wrapper_debug
