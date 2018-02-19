import unittest
import numpy as np

from mdlmc.analysis.rdf import calculate_rdf
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


class TestRDF(unittest.TestCase):
    def test_rdf_single_atomtype(self):
        """Assert that RDF is approximately 1 for uniformly distributed atoms"""
        np.random.seed(0)
        box_length = 14
        pbc = np.array([box_length] * 3, dtype=float)
        atombox = AtomBoxCubic(pbc)

        trajectory = np.random.uniform(0, 100, size=(100, 100, 3))
        selection = np.ones(trajectory.shape[1], dtype=bool)
        rdf, dists = calculate_rdf(trajectory, selection, selection, atombox, 1, box_length/2, 20)
        self.assertAlmostEqual(rdf.mean(), 1, delta=0.1)

    def test_rdf_two_atomtypes(self):
        """Assert that RDF is approximately 1 for two uniformly distributed atom types"""
        box_length = 14
        pbc = np.array([box_length] * 3, dtype=float)
        atombox = AtomBoxCubic(pbc)

        trajectory1 = np.random.uniform(0, 15, size=(300, 100, 3))
        selection1 = np.zeros(100, dtype=bool)
        selection1[range(50)] = 1
        selection2 = np.zeros(100, dtype=bool)
        selection2[range(50, 100)] = 1
        rdf, dists = calculate_rdf(trajectory1, selection1, selection2, atombox,
                                   dmin=1, dmax=box_length/2, bins=22, verbose=True)

        self.assertAlmostEqual(rdf.mean(), 1, delta=0.1)
