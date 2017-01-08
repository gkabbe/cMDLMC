import unittest
from itertools import product

import numpy as np

from mdlmc.KMC import excess_kmc


class TestKMC(unittest.TestCase):

    def test_fastforward_to_next_jump(self):
        """Compare the KMC with time-dependent rates with the KMC with constant rates.
        If the time-dependent KMC scheme gets the same rates at each time step, the
        result must be the same as for the constant rate KMC."""
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
                delta_frame, delta_t = excess_kmc.fastforward_to_next_jump(probsums, 0, dt, frame,
                                                                           time, 1000)
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
                self.assertAlmostEqual(t1, t2, msg="Time-dependent and time-independent KMC do not"
                                                   "deliver the same result!")
                # Assert agreement of KMC time and MD time
                self.assertEqual(int(t1 // dt), s)

    def test_variable_rates(self):
        """Make sinus-like jump rates and check average."""
        timesteps = np.linspace(0, 200 * np.pi, 10000)[:, None]
        probsums = 0.06 + 0.02 * np.sin(timesteps)
        average = np.mean(probsums)
        print(average)
        dt = 0.5
        counter = 0
        frame, time = 0, 0
        for i in range(100000):
            delta_frame, delta_t = excess_kmc.fastforward_to_next_jump(probsums, 0, dt, frame, time,
                                                                       timesteps.shape[0])
            frame = (frame + delta_frame) % timesteps.shape[0]
            time += delta_t
            counter += 1
        jumprate_avg = counter / time
        print("Number of jumps:", counter)
        print("Jump rate:", jumprate_avg)
        relative_error = (jumprate_avg - average) / average
        self.assertLessEqual(relative_error, 0.01, "The average jump rate deviates by more than"
                                                   "one percent!")
