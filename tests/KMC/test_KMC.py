import unittest
from itertools import product

import numpy as np

from mdlmc.KMC import excess_kmc


class TestKMC(unittest.TestCase):

    def test_fastforward_to_next_jump(self):
        """Compare the KMC with time-dependent rates with the KMC with constant rates"""
        # Test for a selection of fixed rates (unit: fs^{-1})
        omegas = [0.03, 0.06, 0.13]
        dts = [0.1, 0.5, 1.3]  # fs

        # Test for different MD time steps
        for dt, omega in product(dts, omegas):
            probsums = np.ones((1000, 1)) * omega
            sweep, frame, time = 0, 0, 0
            times = []
            sweeps = []
            np.random.seed(0)
            for i in range(100):
                delta_frame, delta_t = excess_kmc.ffjn(probsums, 0, dt, frame, time, 1000)
                frame = (frame + delta_frame) % 100
                sweep += delta_frame
                time += delta_t
                times.append(time)
                sweeps.append(sweep)

            time = 0
            times_verify = []
            np.random.seed(0)
            for i in range(100):
                delta_t = -np.log(1 - np.random.random()) / omega
                time += delta_t
                times_verify.append(time)

            print("KMC with variable rates, KMC with fixed rates, Trajectory steps")
            for i, (t1, t2, s) in enumerate(zip(times, times_verify, sweeps)):
                print(t1, t2, s)
                # Assert equal times of both KMC methods
                self.assertAlmostEqual(t1, t2)
                # Assert agreement of KMC time and MD time
                self.assertEqual(int(t1 // dt), s)
