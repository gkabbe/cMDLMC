import unittest

import numpy as np

from mdlmc.KMC import excess_kmc


class TestKMC(unittest.TestCase):

    def test_fastforward_to_next_jump(self):
        """Compare the KMC with time-dependent rates with the KMC with constant rates"""
        # Set a fixed rate of 0.06 / fs
        omega = 0.06
        probsums = np.ones((1000, 1)) * omega
        sweep, frame, time = 0, 0, 0
        jumps = 0
        dt = 0.5  # fs
        times = []
        sweeps = []
        np.random.seed(0)
        for i in range(100):
            delta_frame, delta_t = excess_kmc.ffjn(probsums, 0, dt, frame, time, 1000)
            frame = (frame + delta_frame) % 100
            sweep += delta_frame
            time += delta_t
            jumps += 1
            times.append(time)
            sweeps.append(sweep)

        time = 0
        times_verify = []
        np.random.seed(0)
        for i in range(100):
            delta_t = -np.log(1 - np.random.random()) / omega
            time += delta_t
            times_verify.append(time)

        for i, (t1, t2, s) in enumerate(zip(times, times_verify, sweeps)):
            print(i, t1, t2, s * dt)
            self.assertAlmostEqual(t1, t2)
