import unittest
from unittest.mock import MagicMock, patch, mock_open
from io import StringIO
import numpy as np

from mdlmc.LMC.MDMC import MDMC


fake_output = StringIO()


class FakeMDMC(MDMC):
    def __init__(self):
        self.oxygen_trajectory = np.random.random((1, 20, 3))
        self.oxygennumber = 20
        self.proton_number = 10


class TestMDMC(unittest.TestCase):

    def test_initialize_oxygen_lattice(self):
        mdmc = FakeMDMC()
        oxy_lattice = mdmc.initialize_oxygen_lattice((1, 1, 1))
        self.assertEqual((oxy_lattice > 0).sum(), mdmc.proton_number)
        self.assertTrue(oxy_lattice.sum() == int(mdmc.proton_number * (mdmc.proton_number + 1) / 2))


    def test_md_lmc_run(self):
        pass