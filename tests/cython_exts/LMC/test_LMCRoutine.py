import unittest

import numpy as np

from mdlmc.cython_exts.LMC import LMCHelper
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.cython_exts.LMC.LMCHelper import JumprateFunction, AngleCutoff

class TestLMCRoutine(unittest.TestCase):

    def test_angle_cutoff(self):
        cutoff_angle = np.pi / 2

        angle_fct = AngleCutoff(cutoff_angle)

        for theta in np.linspace(np.pi, 0, 50):
            print("{:.2f} * pi ->".format(theta / np.pi), angle_fct.evaluate(theta))
            self.assertEqual(theta > 0.5 * np.pi, angle_fct.evaluate(theta))

    def test_store_jumprates(self):
        class IdentityFct(JumprateFunction):
            def evaluate(self, distance):
                return distance

        # Initialize atoms that the POO angle is at 90 degrees
        oxy_traj = np.array([[0, 0, 0],
                             [1, 0, 0]], dtype=float).reshape((1, 2, 3))

        phos_traj = np.array([0, 1, 0], dtype=float).reshape((1, 1, 3))

        atombox = AtomBoxCubic([10, 10, 10])
        jumprate_fct = IdentityFct()
        lmc = LMCHelper.LMCRoutine(oxy_traj, phos_traj, atom_box=atombox, jumprate_fct=jumprate_fct,
                                   cutoff_radius=4.0, angle_threshold=np.pi/3,
                                   neighbor_search_radius=4.0, seed=0, verbose=True,
                                   angle_dependency=True)
        lmc.store_jumprates(verbose=True)
        probs = lmc.jump_probability[0]
        # angle PO_1O_2 = 90 -> prob = 1
        # angle PO_2O_1 = 45 -> prob = 0
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[1], 0.0)

