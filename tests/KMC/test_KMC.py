import unittest
from itertools import product, cycle

import numpy as np

from mdlmc.KMC import excess_kmc


class TestVariableJumpRateKMC(unittest.TestCase):

    def test_fastforward_to_next_jump(self):
        """Compare the KMC with time-dependent rates with the KMC with constant rates.
        If the time-dependent KMC scheme gets the same rates at each time step, the
        result must be the same as for the constant rate KMC."""
        # Test for a selection of fixed rates (unit: fs^{-1})
        omegas = [0.03, 0.06, 0.13]
        dts = [0.1, 0.5, 1.3]  # fs

        # Test for different MD time steps
        for dt, omega in product(dts, omegas):
            def probsum_gen(omega):
                while True:
                    yield omega

            jump_rates = probsum_gen(omega)
            fastforward_gen = excess_kmc.fastforward_to_next_jump(jump_rates, dt)
            sweep, frame, time = 0, 0, 0
            times = []
            sweeps = []
            np.random.seed(0)
            for i in range(100):
                frame, delta_frame, kmc_time = next(fastforward_gen)
                sweep += delta_frame
                times.append(kmc_time)
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

    def test_variable_rates_average(self):
        """Make sinus-like jump rates and check average jump rate."""
        timesteps = np.linspace(0, 200 * np.pi, 10000)[:, None]
        jump_rates = 0.06 + 0.02 * np.sin(timesteps)
        average = np.mean(jump_rates)
        jump_rates = cycle(jump_rates)

        print(average)
        dt = 0.5
        counter = 0
        fastforward_gen = excess_kmc.fastforward_to_next_jump(jump_rates, dt)
        for i in range(10000):
            frame, delta_frame, kmc_time = next(fastforward_gen)
            counter += 1
        jumprate_avg = counter / kmc_time
        print("Number of jumps:", counter)
        print("Jump rate:", jumprate_avg)
        relative_error = (jumprate_avg - average) / average
        self.assertLessEqual(relative_error, 0.01, "The average jump rate deviates by more than"
                                                   "one percent!")

    def test_variable_rates_index(self):
        """Use variable rates that are all zero except for one entry.
        Make sure that the jump always happens at that index"""
        jumprate_length = 117
        nonzero_index = 73
        jump_rates = np.zeros(jumprate_length)
        jump_rates[nonzero_index] = 0.17
        jump_rates = cycle(jump_rates)
        dt = 0.22

        fastforward_gen = excess_kmc.fastforward_to_next_jump(jump_rates, dt)

        for i, (frame, delta_frame, kmc_time) in enumerate(fastforward_gen):
            print(frame % jumprate_length, delta_frame / jumprate_length)
            if i > 0:
                self.assertEqual(frame % jumprate_length, nonzero_index)
            if i == 100:
                break


class TestGenerators(unittest.TestCase):
    def test_trajectory_generator(self):
        """Assert that the trajectory generator correctly yields the trajectory frames and
        does repeat itself after a complete iteration.
        Also check that the returned counter counts correctly"""
        trajectory = np.zeros((5, 1, 3))
        trajectory[:, 0, 0] = range(5)

        trajgen = excess_kmc.trajectory_generator(trajectory)

        for i in range(5):
            counter, dist = next(trajgen)
            self.assertEqual(i, dist[0, 0])
            self.assertEqual(counter, i)
        for i in range(5):
            counter, dist = next(trajgen)
            self.assertEqual(counter, i + 5)
            self.assertEqual(i, dist[0, 0])

    def test_distance_generator(self):
        """Assert that the distances are correctly generated"""
        x = 10
        x_rescaled = 5
        distances = np.zeros((5, 1, 3))
        distances[:, 0, 0] = x
        distances_rescaled = np.zeros((5, 1, 3))
        distances_rescaled[:, 0, 0] = x_rescaled
        relaxation_time = 6

        def some_fct(x):
            return None

        params = ()

        kmcgen = excess_kmc.KMCGen(oxy_idx=0, distances=distances,
                                   distances_rescaled=distances_rescaled, jumprate_fct=some_fct,
                                   jumprate_params=params)

        distance_gen = kmcgen.distance_generator()

        # Assert that distance_gen returns the rescaled distances if delay is not set
        dist = next(distance_gen)
        self.assertTrue(np.allclose(dist, [x_rescaled, 0, 0]))

        # Assert that the delay results in larger distances, which will be scaled linearly down
        # to the scaled distances
        kmcgen.relaxation_time = relaxation_time
        for i in range(6):
            x_generated, *_ = next(distance_gen)
            self.assertAlmostEqual(x_generated, x - i / relaxation_time * (x - x_rescaled),
                               "The unrescaled distance should be greater than the rescaled distance")

        # Assert that after the delay has passed, the distances are equal to the rescaled
        # distances
        for i in range(20):
            x_generated, *_ = next(distance_gen)
            self.assertEqual(x_generated, x_rescaled)

    def test_distance_generator_reset(self):
        """Assert bla"""
        x = 10
        x_rescaled = 5

        distances = np.zeros((5, 1, 3))
        distances[:, 0, 0] = x
        distances_rescaled = np.zeros((5, 1, 3))
        distances_rescaled[:, 0, 0] = x_rescaled
        relaxation_time = 10

        def some_fct(x):
            return None

        params = ()

        kmcgen = excess_kmc.KMCGen(oxy_idx=0, distances=distances,
                                   distances_rescaled=distances_rescaled, jumprate_fct=some_fct,
                                   jumprate_params=params)

        distance_gen = kmcgen.distance_generator()

        # Generate some distances without relaxation time set
        for i in range(5):
            next(distance_gen)

        # Set relaxation time
        kmcgen.relaxation_time = relaxation_time
        for i in range(6):
            next(distance_gen)

        # Now reset the relaxation time and make sure everything works as expected
        kmcgen.reset_relaxationtime(10)
        for i in range(6):
            x_generated, *_ = next(distance_gen)
            print(x_generated, x - i / relaxation_time * (x - x_rescaled))
            self.assertEqual(x_generated, x - i / relaxation_time * (x - x_rescaled))

    def test_jumprate_generator(self):

        x = 10
        x_rescaled = 5
        relaxation_time = 20

        distances = np.zeros((5, 1, 3))
        distances[:, 0, 0] = x
        distances_rescaled = np.zeros((5, 1, 3))
        distances_rescaled[:, 0, 0] = x_rescaled

        def some_fct(x):
            return x

        params = ()

        kmcgen = excess_kmc.KMCGen(oxy_idx=0, distances=distances,
                                   distances_rescaled=distances_rescaled, jumprate_fct=some_fct,
                                   jumprate_params=params)

        probsum_gen = kmcgen.jumprate_generator()
        for frame, probsum in enumerate(probsum_gen):
            self.assertEqual(probsum, x_rescaled)
            if frame == 20:
                break

        probsum_gen = kmcgen.jumprate_generator()
        kmcgen.relaxation_time = relaxation_time
        for frame, probsum in enumerate(probsum_gen):
            self.assertEqual(probsum, x - frame / relaxation_time * (x - x_rescaled))
            print(frame, probsum)
            if frame == 20:
                break


class TestPositionTracker(unittest.TestCase):
    def setUp(self):
        class MockAtomBox:
            def distance(self, pos1, pos2):
                return pos2 - pos1
        self.atombox = MockAtomBox()

    def test_linear_chain(self):

        d_oo = 2.6
        d_oh = 0.95  # unit: angstrom
        proton_idx = 0
        chain_length = 20
        trajectory = np.zeros((1, chain_length, 3))
        trajectory[0, :, 0] = np.linspace(0, (chain_length - 1) * d_oo, chain_length)
        position_tracker = excess_kmc.PositionTracker(trajectory, self.atombox, proton_idx, d_oh)

        for i in range(1, chain_length):
            # Proton walks from left to right through trajectory
            new_proton_idx = (proton_idx + 1) % trajectory.shape[1]
            position_tracker.update_correction_vector(frame_idx=0, new_proton_position=new_proton_idx)
            print("Correction vector:", position_tracker.correction_vector)
            proton_pos = position_tracker.get_position(0)
            proton_idx = new_proton_idx
            desired = np.array([i * (d_oo - 2 * d_oh), 0, 0])
            print("Position:", proton_pos)
            print("Desired:", desired)
            np.testing.assert_allclose(proton_pos, desired)

    def test_water_chain(self):
        """Make a chain of perfectly arranged water molecules.
        One OH vector of each water molecule points to the "back" of the next water.
        In contrast to the linear chain, this introduces an additional y-component.
        """
        # Construct the water chain
        chain_length = 20
        angle_oho = 104.45
        angle_horizontal = angle_oho / 4
        dist_oh = 0.95
        dist_oh_horizontal = dist_oh * np.cos(np.radians(angle_horizontal))
        dist_oh_vertical = dist_oh * np.sin(np.radians(angle_horizontal))
        dist_oo = 2.6
        dist_oo_horizontal = dist_oo * np.cos(np.radians(angle_horizontal))
        dist_oo_vertical = dist_oo * np.sin(np.radians(angle_horizontal))
        oxygens = np.zeros((1, 20, 3))
        oxygens[0, :, 0] = np.linspace(0, dist_oo_horizontal * (chain_length - 1), chain_length)
        oxygens[0, 1::2, 1] = dist_oo_vertical

        desired = np.zeros(3)

        proton_idx = 0
        position_tracker = excess_kmc.PositionTracker(oxygens, self.atombox, proton_idx, dist_oh)

        for i in range(1, chain_length):
            new_proton_idx = (proton_idx + 1) % oxygens.shape[1]
            position_tracker.update_correction_vector(0, new_proton_idx)
            print("Correction vector:", position_tracker.correction_vector)
            proton_pos = position_tracker.get_position(0)
            proton_idx = new_proton_idx
            desired += [dist_oo_horizontal - 2 * dist_oh_horizontal, 0, 0]
            if i % 2 == 1:
                desired[1] = dist_oo_vertical - 2 * dist_oh_vertical
            else:
                desired[1] = 0
            print("Position:", proton_pos)
            print("Desired:", desired)
            np.testing.assert_allclose(proton_pos, desired, atol=1e-7)
