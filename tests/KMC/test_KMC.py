import unittest

import numpy as np

from mdlmc.KMC import excess_kmc


class TestKMC(unittest.TestCase):

    def test_fastforward_to_next_jump(self):
        """Compare the KMC with time-dependent rates with the KMC with constant rates"""
        # Set a fixed rate of 0.06 / fs
        omega = 0.06
        probsums = np.ones((1000, 2)) * omega
        probsums[:, 1] = range(1000)
        sweep, frame, time = 0, 0, 0
        jumps = 0
        dt = 0.5  # fs
        times = []
        np.random.seed(0)
        for i in range(100):
            delta_frame, delta_t = excess_kmc.fastforward_to_next_jump(probsums, 0, dt, frame, time)
            frame = (frame + delta_frame) % 100
            sweep += delta_frame
            time += delta_t
            jumps += 1
            times.append(time)

        time = 0
        times_verify = []
        np.random.seed(0)
        for i in range(100):
            delta_t = -np.log(np.random.random()) / omega
            time += delta_t
            times_verify.append(time)

        for t1, t2 in zip(times, times_verify):
            print(t1, t2)
