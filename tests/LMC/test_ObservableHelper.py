import unittest
from unittest.mock import MagicMock

import numpy as np
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import ObservableManager


class TestObservableHelper(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        pass

    def test_calculate_displacement_cubic(self):
        """Assert correct displacement calculation in cubic box"""
        helper = MagicMock()
        helper.get_jumps.return_value = 42
        atombox = AtomBoxCubic([4, 5, 8])
        oxygen_trajectory = np.zeros((1, 20, 3))
        oxygen_trajectory[0, :, 1] = np.linspace(0, 19, 20)
        oxygen_lattice = np.zeros(20, dtype=int)
        oxygen_lattice[0] = 1
        obsman = ObservableManager(helper, oxygen_trajectory, atombox, oxygen_lattice,
                                   proton_number=1, md_timestep=0.5, sweeps=1000)

        pos = 0
        for i in range(30):
            print("snapshot before:", obsman.proton_pos_snapshot)
            oxygen_lattice[pos] = 0
            pos = (pos + 1) % 20
            oxygen_lattice[pos] = 1
            obsman.calculate_displacement(0)
            print(obsman.displacement)
            print(obsman.displacement == [0, 1.0, 0])
            assert (obsman.displacement == [0, i + 1, 0]).all()
            print("snapshot afterwards:", obsman.proton_pos_snapshot)
            print("displacement:", obsman.displacement)
