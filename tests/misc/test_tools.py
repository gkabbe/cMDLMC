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

    def test_online_variance_generator(self):
        data = np.random.uniform(-5, 5, size=1000)

        vargen = online_variance_generator()

        for i, x in enumerate(data):
            next(vargen)
            gen_var = vargen.send(x)
        np_var = data.var()
        self.assertLess((gen_var - np_var) / np_var, 0.01, "Online variance deviates by more than"
                                                           "one percent from np.var")

        # Test it on arrays
        data = np.random.uniform(-5, 5, size=(1000, 100))
        vargen = online_variance_generator(data_size=data.shape[1])

        for x in data:
            next(vargen)
            gen_var = vargen.send(x)
        np_var = data.var(axis=0)
        self.assertTrue(((gen_var - np_var) / np_var < 0.01).all(),
                        "Online variance deviates by more than one percent from np.var")

        # Test on arrays with mask

        vargen = online_variance_generator(data_size=data.shape[1], use_mask=True)
        mask = np.zeros(data.shape[1], dtype=bool)
        mask[0] = 1

        for x in data:
            next(vargen)
            vargen.send(x)
            gen_var = vargen.send(mask)

        np_var = np.take(data, 0, axis=1).var()
        self.assertTrue(((gen_var[0] - np_var) / np_var < 0.01).all(),
                        "Online variance deviates by more than one percent from np.var")
