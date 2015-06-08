#!/usr/bin/python

import unittest
import numpy as np
import ipdb

from cython_exts.kMC import kMC_helper as kh
from cython_exts.atoms import numpyatom as npa
from IO import BinDump
from kMC import MDMC

class TestMDMC(unittest.TestCase):
    def test_extend_simulationbox(self):
        fermi_params = np.array([1, 3.5, 1e-50])
        pbc = np.array([303., 303., 303.])
        h = np.zeros((3,3))
        h[0,0] = pbc[0]
        h[1,1] = pbc[1]
        h[2,2] = pbc[2]
        mdmc = MDMC.MDMC()
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
            helper = kh.Helper(pbc_extended, nonortho=0, jumprate_parameters=fermi_params, verbose=True)
            mdmc.extend_simulationbox(Os, 101, h, boxmult, nonortho=False)
            helper.determine_neighbors(Os, 6.)
            helper.calculate_transitions_new(Os, fermi_params, 20.)
            start, dest, prob = helper.return_transitions()
            sdp = zip(start, dest, prob)
            transition_matrix = np.zeros((Os.shape[0], Os.shape[0]))
            matrix_as_it_should_be = np.eye(Os.shape[0], k=1)+np.eye(Os.shape[0], k=-1)
            matrix_as_it_should_be[0,-1] = 1.0
            matrix_as_it_should_be[-1,0] = 1.0
            for s,d,p in sdp:
                transition_matrix[s,d] = p
            #Check if Transitions matrix looks as expected (each atom connected to its predecessor and successor)
            self.assertTrue(np.allclose(transition_matrix, matrix_as_it_should_be), "Extending in direction {} failed".format(direction))

if __name__ == "__main__":
    unittest.main()
