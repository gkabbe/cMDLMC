import unittest
import numpy as np

from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic

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

        self.assertAlmostEqual(self.atombox_cubic.angle(atom_1, atom_2, atom_3), np.pi / 2)

    def test_atomboxcubic_nextneighbor(self):
        pbc = np.asfarray([100, 100, 100])
        atoms = np.random.uniform(0.3, 50, size=(20, 3))
        atombox = AtomBoxCubic(pbc)

        for i in range(10):
            atom = np.random.uniform(0, 50, size=3)
            index, distance = atombox.next_neighbor(atom, atoms)
            index_cmp = np.argmin(np.sqrt(((atom - atoms)**2).sum(axis=-1)))
            self.assertEqual(index, index_cmp)

    def test_atomboxcubic_position_extended_box(self):
        pbc = np.asfarray([10, 10, 10])
        atom1 = np.asfarray([[0, 0, 0]])

        box_multiplier = (1, 1, 10)
        atombox = AtomBoxCubic(pbc, box_multiplier=box_multiplier)

        for i in range(10):
            pos = atombox.position_extended_box(i, atom1)
            pos_target = np.asfarray([0, 0, 10]) * i
            print(pos, pos_target)
            self.assertTrue(np.allclose(pos, pos_target))

        box_multiplier = (1, 10, 1)
        atombox = AtomBoxCubic(pbc, box_multiplier=box_multiplier)

        for i in range(10):
            pos = atombox.position_extended_box(i, atom1)
            pos_target = np.asfarray([0, 10, 0]) * i
            print(pos, pos_target)
            self.assertTrue(np.allclose(pos, pos_target))

        box_multiplier = (10, 1, 1)
        atombox = AtomBoxCubic(pbc, box_multiplier=box_multiplier)

        for i in range(10):
            pos = atombox.position_extended_box(i, atom1)
            pos_target = np.asfarray([10, 0, 0]) * i
            print(pos, pos_target)
            self.assertTrue(np.allclose(pos, pos_target))

        box_multiplier = (5, 5, 5)
        atombox = AtomBoxCubic(pbc, box_multiplier=box_multiplier)

        index = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    pos = atombox.position_extended_box(index, atom1)
                    pos_target = i * np.asfarray([10, 0, 0]) + j * np.asfarray(
                        [0, 10, 0]) + k * np.asfarray([0, 0, 10])
                    print(pos, pos_target)
                    self.assertTrue(np.allclose(pos, pos_target))
                    index += 1

    def test_atomboxcubic_nextneighbor_extended_box(self):
        """Test that """
        pbc = np.asfarray([10, 10, 10])
        atoms_1 = np.asfarray([[0, 0, 0]])
        atoms_2 = np.asfarray([[0, 0, 1]])

        box_multipliers = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2),
                           (2, 2, 2), (2, 2, 2), (2, 2, 2)]

        indices = [0, 1, 1, 1, 2, 2, 3, 4, 5]

        for bm, index in zip(box_multipliers, indices):
            atombox = AtomBoxCubic(pbc, box_multiplier=bm)
            print(atombox.next_neighbor_extended_box(index, atoms_1, atoms_2))

        box_multiplier = (1, 1, 5)
        atoms_1 = np.asfarray([[0, 0, 0]])
        atoms_2 = np.asfarray([[0, 0, 9]])
        atombox = AtomBoxCubic(pbc, box_multiplier=box_multiplier)
        print(atombox.next_neighbor_extended_box(0, atoms_1, atoms_2))

    def test_compare_cubic_and_monoclinic(self):
        """"Make sure both boxes give the same result for same box vectors"""
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

    def test_length_all_to_all(self):
        pbc = np.asfarray([10, 10, 10])
        atombox = AtomBoxCubic(pbc)
        atoms = np.asfarray([[0, 0, 0],
                             [1, 1, 1],
                             [5, 5, 5],
                             [10, 10, 10]])
        distances = atombox.length_all_to_all(atoms, atoms)
        desired_result = np.asfarray([[0, np.sqrt(3), np.sqrt(3 * 5**2), 0],
                                      [np.sqrt(3), 0, np.sqrt(3 * 4**2), np.sqrt(3)],
                                      [np.sqrt(3 * 5**2), np.sqrt(3 * 4**2), 0, np.sqrt(3 * 5**2)],
                                      [0, np.sqrt(3), np.sqrt(3 * 5**2), 0]
                                     ])
        np.testing.assert_allclose(distances, desired_result)
