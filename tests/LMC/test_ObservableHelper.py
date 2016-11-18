import unittest
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic


class TestObservableHelper(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        pass

    def test_calculate_displacement(self):
        helper = MagicMock()
        helper.get_jumps.return_value = 42
        atombox = AtomBoxCubic([10, 20, 30])
        oxygen_trajectory = np.zeros((20, 3))
        oxygen_trajectory[:, 1] = np.linspace(0, 19, 20)
        oxygen_lattice = np.zeros(20, dtype=int)
        print(oxygen_trajectory)
        obsman = ObservableManager(helper, oxygen_trajectory, atombox, oxygen_lattice,
                                   proton_number=1, md_timestep=0.5, sweeps=1000)
