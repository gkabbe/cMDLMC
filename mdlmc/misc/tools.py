from functools import wraps
import time
import argparse
import inspect


def chunk(iterable, step):
    starts = range(0, len(iterable) - 1, step)
    stops = map(lambda x: min(x, len(iterable)), range(step, len(iterable) + step, step))
    for start, stop in zip(starts, stops):
        yield start, stop


def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        total_time = time.time() - start_time
        print("# Total time for {}: {:.2f}".format(f.__name__, total_time))
        return result

    return wrapper


def argparse_compatible(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and type(args[0]) == argparse.Namespace:
            expected_args = inspect.signature(func).parameters.keys()
            args = {arg: getattr(args[0], arg) for arg in expected_args if hasattr(args[0], arg)}
            return func(**args)
        else:
            return func(*args, **kwargs)

    return wrapper
