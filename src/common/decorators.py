"""Decorators for the repo"""

import functools
import time
import logging
from typing import Any, Callable, TypeVar

RT = TypeVar("RT")  # return type


def timer(func: Callable[..., RT]) -> Callable[..., RT]:
    """Compute and print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args: Any, **kwargs: Any) -> RT:
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.debug(
            "Finished running %r function in %.4f secs", func.__name__, run_time
        )
        return value

    return wrapper_timer


def debug(func: Callable[..., RT]) -> Callable[..., RT]:
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args: Any, **kwargs: Any) -> RT:  # --enable-incomplete-features
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug("Calling %s  %s", func.__name__, signature)
        value = func(*args, **kwargs)
        logging.debug("%r returned %r", func.__name__, value)
        return value

    return wrapper_debug
