from functools import wraps
import time
import argparse
import inspect
import pickle
import os

import numpy as np


def chunk(iterable, step, length=None):
    assert step > 0
    if not length:
        length = len(iterable)

    starts = range(0, length, step)
    stops = map(lambda x: min(x, length), range(step, length + step, step))
    for start, stop in zip(starts, stops):
        yield start, stop, iterable[start: stop]


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


def remember_results(overwrite=False, nobackup=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nobackup_str = "_nobackup" if nobackup else ""
            save_fname = func.__name__ + nobackup_str + "_result.pickle"
            key = tuple(args) + tuple(sorted(kwargs.items()))
            if os.path.exists(save_fname):
                with open(save_fname, "rb") as f:
                    results_dict = pickle.load(f)
                results = results_dict[key]
            else:
                results_dict = dict()
            if overwrite or key not in results_dict:
                results = func(*args, **kwargs)
                results_dict[key] = results

            with open(save_fname, "wb") as f:
                pickle.dump(results_dict, f)

            return results
        return wrapper
    return decorator


def read_file(file, *, usecols=None, dtype=float):
    arr_length = 0
    for line in file:
        if not line.lstrip().startswith("#"):
            arr_length += 1
    file.seek(0)
    print("File length:", arr_length)

    if usecols:
        arr_width = len(usecols)
        usecols = list(usecols)
    else:
        line = file.readline().lstrip()
        while line.startswith("#"):
            line = file.readline().lstrip()
        arr_width = len(line.split())
        usecols = list(range(arr_width))

    print("Using columns", usecols)
    print("Creating array of shape", (arr_length, arr_width))

    array = np.zeros((arr_length, arr_width), dtype=dtype)

    file.seek(0)

    i = 0
    for line in file:
        if not line.lstrip().startswith("#"):
            try:
                linecontent = np.fromstring(line, sep=" ")
                array[i] = linecontent[usecols]
            except:
                print("error")
                print(linecontent)
                break
            i += 1

    return array
