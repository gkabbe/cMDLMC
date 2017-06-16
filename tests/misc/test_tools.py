from unittest import TestCase

import numpy as np

from mdlmc.misc.tools import chunk, chunk_trajectory, online_variance_generator


def test_chunk():
    simple_range = range(100)
    range_chunk = chunk(simple_range, 3)
    for start, stop, chk in range_chunk:
        assert simple_range[start: stop] == chk

    assert chk[-1] == 99

    some_list = list("abcdefghijklmnopqrstuvwxyz")

    for start, stop, chk in chunk(some_list, 7):
        assert some_list[start: stop] == chk

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for start, stop, chk in chunk(alphabet, 11):
        assert alphabet[start: stop] == chk


def test_chunk_trajectory():
    trajectory = np.array([np.arange(99).reshape((33, 3)) for _ in range(20)])
    selection = np.zeros(33, dtype=bool)
    selection[0] = 1
    selection[-1] = 1
    chunk_gen = chunk_trajectory(trajectory, 3, selection=selection)
    for _, _, chk in chunk_gen:
        assert (chk == np.array([[0, 1, 2], [96, 97, 98]])).all()

    for _, _, chk in chunk_trajectory(trajectory, 3):
        assert (chk == np.arange(99).reshape((33, 3))).all()
