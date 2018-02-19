from unittest import TestCase

import numpy as np

from mdlmc.misc.tools import chunk, chunk_trajectory, online_variance_generator


class TestTools(TestCase):
    def test_chunk(self):

        simple_range = range(100)
        range_chunk = chunk(simple_range, 3)
        for start, stop, chk in range_chunk:
            self.assertEqual(simple_range[start: stop], chk)

        self.assertEqual(chk[-1], 99)

        some_list = list("abcdefghijklmnopqrstuvwxyz")

        for start, stop, chk in chunk(some_list, 7):
            self.assertEqual(some_list[start: stop], chk)

        alphabet = "abcdefghijklmnopqrstuvwxyz"

        for start, stop, chk in chunk(alphabet, 11):
            self.assertEqual(alphabet[start: stop], chk)

    def test_chunk_trajectory(self):
        trajectory = np.array([np.arange(99).reshape((33, 3)) for i in range(20)])

        selection = np.zeros(33, dtype=bool)
        selection[0] = 1
        selection[-1] = 1

        chunk_gen = chunk_trajectory(trajectory, 3, selection=selection)

        for _, _, chk in chunk_gen:
            self.assertTrue((chk == np.array([[0, 1, 2], [96, 97, 98]])).all())

        for _, _, chk in chunk_trajectory(trajectory, 3):
            self.assertTrue((chk == np.arange(99).reshape((33, 3))).all())
