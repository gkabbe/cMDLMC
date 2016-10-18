import unittest
import numpy as np

from mdlmc.analysis.rdf import distance_histogram, calculate_rdf
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


class TestRDF(unittest.TestCase):
    def test_rdf_single_atomtype(self):
        """Assert that RDF is approximately 1 for uniformly distributed atoms"""
        np.random.seed(0)
        pbc = np.array([14, 14, 14], dtype=float)
        atombox = AtomBoxCubic(pbc)

        trajectory = np.random.uniform(0, 100, size=(100, 100, 3))
        rdf, dists = calculate_rdf([trajectory], atombox, {"bins": 50, "range": (0, 7)})
        self.assertAlmostEqual(rdf.mean(), 1, delta=0.1)

    def test_rdf_two_atomtypes(self):
        """Assert that RDF is approximately 1 for two uniformly distributed atom types"""
        pbc = np.array([14, 14, 14], dtype=float)
        atombox = AtomBoxCubic(pbc)

        trajectory1 = np.random.uniform(0, 100, size=(100, 50, 3))
        trajectory2 = np.random.uniform(0, 100, size=(100, 50, 3))
        rdf, dists = calculate_rdf([trajectory1, trajectory2], atombox,
                                   {"bins": 50, "range": (0, 7)})

        self.assertAlmostEqual(rdf.mean(), 1, delta=0.1)
