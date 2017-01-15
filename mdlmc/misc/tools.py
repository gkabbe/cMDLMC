from functools import wraps
import time
import argparse
import inspect
import pickle
import os

import numpy as np


def chunk(iterable, chunk_size, length=None):
    assert chunk_size > 0
    if not length:
        length = len(iterable)

    starts = range(0, length, chunk_size)
    stops = map(lambda x: min(x, length), range(chunk_size, length + chunk_size, chunk_size))
    for start, stop in zip(starts, stops):
        yield start, stop, iterable[start:stop]


def chunk_trajectory(trajectory, chunk_size, length=None, selection=None):
    """Chunk Numpy/HDF5 trajectory
    Parameters
    ----------
    trajectory: array_like
        The trajectory that will be iterated in chunks.
        Expected dimensions: (no_of_frames, no_of_atoms, 3)
    chunk_size: int
        Chunk size
    length: int
        Trajectory length.
        Allows iterating over only a part of the trajectory
    selection: array_like
        Expects a boolean array"""

    assert chunk_size > 0, "Chunk size must be greater than zero"
    if not length:
        length = trajectory.shape[0]
    starts = range(0, length, chunk_size)
    stops = map(lambda x: min(x, length), range(chunk_size, length + chunk_size, chunk_size))

    if selection is None:
        selection = slice(None)

    for start, stop in zip(starts, stops):
        yield start, stop, trajectory[start:stop, selection]


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
            func_params = inspect.signature(func).parameters
            expected_args = func_params.keys()
            args_dict = {}
            for arg in expected_args:
                if hasattr(args[0], arg):
                    args_dict[arg] = getattr(args[0], arg)
                elif func_params[arg].default is not inspect._empty:
                    args_dict[arg] = func_params[arg].default
            return func(**args_dict)
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


def read_file(file, *, usecols=None, dtype=float, memmap_file=None, shape=None, chunks=1000,
              verbose=False):

    def data_gen(file, chunks):
        """Generator that iterates over <chunks> lines of a file"""
        for i, line in enumerate(file):
            if i == chunks:
                break
            yield line

    if not shape:
        if verbose:
            print("# No shape specified")
            print("# Determine shape by reading file once")

        arr_length = 0
        for line in file:
            if not line.lstrip().startswith("#"):
                arr_length += 1
        file.seek(0)
        if verbose:
            print("# File length:", arr_length)

        if usecols:
            arr_width = len(usecols)
            usecols = list(usecols)
        else:
            line = file.readline().lstrip()
            while line.startswith("#"):
                line = file.readline().lstrip()
            arr_width = len(line.split())
            usecols = list(range(arr_width))

        shape = (arr_length, arr_width)

    print("# Using columns", usecols)
    print("# Creating array of shape", shape, flush=True)

    if memmap_file:
        array = np.memmap(memmap_file, dtype=dtype, shape=shape, mode="w+")
    else:
        array = np.zeros(shape, dtype=dtype)

    file.seek(0)

    position = 0

    while True:
        if verbose:
            print("# {: 7d} / {: 7d}".format(position, shape[0]), end="\r", flush=True)
        tmp_array = np.genfromtxt(data_gen(file, chunks), usecols=usecols)
        if tmp_array.shape[0] == 0:
            break
        array[position: position + tmp_array.shape[0]] = tmp_array
        position += tmp_array.shape[0]

    if memmap_file:
        array.flush()

    return array
