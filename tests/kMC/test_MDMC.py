#!/usr/bin/env python3

import unittest
import numpy as np
import ipdb

from mdkmc.cython_exts.kMC import kMC_helper as kh
from mdkmc.kMC import MDMC

class TestMDMC(unittest.TestCase):
    def test_extend_simulationbox(self):
        fermi_params = dict(a=1.0, b=3.5, c=1e-50)
        pbc = np.array([303., 303., 303.])
        h = np.zeros((3,3))
        h[0,0] = pbc[0]
        h[1,1] = pbc[1]
        h[2,2] = pbc[2]
        mdmc = MDMC
        for direction in ["x", "y", "z"]:
            Os = np.zeros((303, 3), float)
            if direction == "x":
                Os [:101, 0] = np.linspace(0, 300, 101)
                boxmult = [3,1,1]
            elif direction == "y":
                Os [:101, 1] = np.linspace(0, 300, 101)
                boxmult = [1,3,1]
            else:
                Os [:101, 2] = np.linspace(0, 300, 101)
                boxmult = [1,1,3]
            pbc_extended = pbc * boxmult
            #choose parameters so jump prob is 1 for d <=3.5 and 0 else
            helper = kh.Helper(pbc_extended, nonortho=0, jumprate_parameter_dict=fermi_params, verbose=True)
            mdmc.extend_simulation_box(Os, 101, h, boxmult, nonortho=False)
            helper.determine_neighbors(Os, 6.)
            helper.calculate_transitions_new(Os, 20.)
            start, dest, prob = helper.return_transitions()
            sdp = list(zip(start, dest, prob))
            transition_matrix = np.zeros((Os.shape[0], Os.shape[0]))
            matrix_as_it_should_be = np.eye(Os.shape[0], k=1)+np.eye(Os.shape[0], k=-1)
            matrix_as_it_should_be[0,-1] = 1.0
            matrix_as_it_should_be[-1,0] = 1.0
            for s,d,p in sdp:
                transition_matrix[s,d] = p
            #Check if Transitions matrix looks as expected (each atom connected to its predecessor and successor)
            self.assertTrue(np.allclose(transition_matrix, matrix_as_it_should_be),
                            "Extending in direction {} failed".format(direction))

    def test_calculate_displacement(self):
        oxy_positions = np.array([[1.0*i, 0, 0] for i in range(10)])
        initial_proton_pos = np.zeros((1,3))
        pbc = np.array([10., 0, 0])
        displacement = np.zeros((1,3))

        # Test unwrapped case
        for i in range(100):
            oxy_lattice = np.zeros(10, int)
            oxy_lattice[i % 10] = 1
            MDMC.calculate_displacement(oxy_lattice, initial_proton_pos,
                                        oxy_positions, displacement, pbc, wrap=False)
            print("Unwrapped displacement:", displacement)
            print(np.linalg.norm(displacement[0]), i)
            self.assertTrue(np.linalg.norm(displacement[0])==i)

        initial_proton_pos[:] = 0
        # Test wrapped case
        for i in range(100):
            oxy_lattice = np.zeros(10, int)
            oxy_lattice[i % 10] = 1
            MDMC.calculate_displacement(oxy_lattice, initial_proton_pos,
                                        oxy_positions, displacement, pbc, wrap=True)
            dist_target = i % 10
            dist_target = min(abs(dist_target), 10-abs(dist_target))
            print("Wrapped displacement:", displacement, dist_target)
            print(np.linalg.norm(displacement[0]), end=' ')
            self.assertTrue(np.allclose(np.linalg.norm(displacement[0]), dist_target),
                            "{} {}".format(np.linalg.norm(displacement[0]), dist_target))


    def test_load_configfile_new(self):
        testfile1 = "config1.cfg"
        config_dict1 = MDMC.load_configfile_new(testfile1)
        print(config_dict1)


    def test_print_confighelp(self):
        MDMC.print_confighelp()


if __name__ == "__main__":
    unittest.main()
