"""Decorators for the decision tree module"""

import functools
import time
import logging


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(
            "Finished running %s function in %.4f secs", func.__name__, run_time
        )
        return value

    return wrapper_timer
