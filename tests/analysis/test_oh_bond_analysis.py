import numpy as np
import unittest

from mdlmc.analysis import oh_bond_analysis


class TestOHBondAutoCorrelation(unittest.TestCase):
    def test_determine_hydronium_indices(self):

        test_array = np.array([[1, 1, 2, 2, 2],
                               [1, 1, 1, 2, 2]])
        result = oh_bond_analysis.determine_hydronium_indices(test_array)
        np.testing.assert_equal(result, [[2], [1]])

    def test_oh_bond_array_filename(self):
        filename = "the_array.npy"

        assert oh_bond_analysis.oh_bond_array_filename(filename) == \
                         "the_array_cbo.npy"

    def test_autocorrelate(self):
        # TODO: Improve test
        test_array = np.ones(100).reshape((-1, 1))

        result = oh_bond_analysis.autocorrelate(test_array, 10, 20, verbose=True)

        assert np.all(result == np.ones(20))
