from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic

import unittest
import numpy as np

np.random.seed(0)


class TestAtomBoxes(unittest.TestCase):
    def setUp(self):
        self.pbc_cubic = np.asfarray([10, 10, 10])
        self.pbc_monoclinic = np.asfarray([10, 0, 0, 0, 10, 0, 0, 0, 10])
        self.atombox_cubic = AtomBoxCubic(self.pbc_cubic)
        self.atombox_monoclinic = AtomBoxMonoclinic(self.pbc_monoclinic)

    def test_atomboxcubic_length(self):

        atom_1 = np.asfarray([0, 0, 0])
        atom_2 = np.asfarray([6, 6, 6])

        desired_result = np.sqrt((np.asfarray([4, 4, 4]) ** 2).sum())

        for i in range(-5, 5):
            a2 = atom_2 + i * 10
            self.assertAlmostEqual(self.atombox_cubic.length(atom_1, a2)[0], desired_result)

        # Test vectorized input

        atoms_1 = np.zeros((20, 3), dtype=float)
        atoms_2 = np.arange(-10, 10)[:, None] * np.asfarray([10, 10, 10]) + 3

        desired_result = np.ones(20) * np.sqrt(27)
        self.assertTrue(
            np.isclose(self.atombox_cubic.length(atoms_1, atoms_2), desired_result).all())

    def test_atomboxcubic_angle(self):
        atom_1 = np.asfarray([0, 0, 0])
        atom_2 = np.asfarray([3, 0, 0])
        atom_3 = np.asfarray([3, 34, 0])

        self.assertAlmostEqual(self.atombox_cubic.angle(atom_1, atom_2, atom_3), np.pi/2)

    def test_atomboxcubic_nextneighbor(self):
        pbc = np.asfarray([100, 100, 100])
        atom = np.zeros(3)
        atoms = np.random.uniform((0.3, 50), size=(20, 3))
        atombox = AtomBoxCubic(pbc)

        index, distance = atombox.next_neighbor()

    def test_compare_cubic_and_monoclinic(self):

        atom_1 = np.random.uniform(-10, 10, size=(10, 3))
        atom_2 = np.random.uniform(-10, 10, size=(10, 3))
        atom_3 = np.random.uniform(-10, 10, size=(10, 3))

        for i in range(10):
            dist_c = self.atombox_cubic.distance(atom_1[i], atom_2[i])
            dist_m = self.atombox_cubic.distance(atom_1[i], atom_2[i])
            len_c, = self.atombox_monoclinic.length(atom_1[i], atom_2[i])
            len_m, = self.atombox_monoclinic.length(atom_1[i], atom_2[i])
            angle_c = self.atombox_cubic.angle(atom_1[i], atom_2[i], atom_3[i])
            angle_m = self.atombox_monoclinic.angle(atom_1[i], atom_2[i], atom_3[i])

            self.assertTrue(np.isclose(dist_c, dist_m).all())
            self.assertAlmostEqual(len_c, len_m)
            self.assertAlmostEqual(angle_c, angle_m)

