# coding=utf-8

import pytest
import numpy as np
from itertools import cycle, product

from mdlmc.LMC import MDMC


def test_fastforward_to_next_jump():
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
        fastforward_gen = MDMC.KMCLattice.fastforward_to_next_jump(jump_rates, dt)
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
            assert round(abs(t1 - t2), 7) == 0, \
                "Time-dependent and time-independent KMC do not deliver the same result!"
            # Assert agreement of KMC time and MD time
            assert int(t1 // dt) == s


def test_variable_rates_average():
    """Make sinus-like jump rates and check average jump rate."""
    timesteps = np.linspace(0, 200 * np.pi, 10000)[:, None]
    jump_rates = 0.06 + 0.02 * np.sin(timesteps)
    average = np.mean(jump_rates)
    jump_rates = cycle(jump_rates)

    print(average)
    dt = 0.5
    counter = 0
    fastforward_gen = MDMC.KMCLattice.fastforward_to_next_jump(jump_rates, dt)
    for i in range(10000):
        frame, delta_frame, kmc_time = next(fastforward_gen)
        counter += 1
    jumprate_avg = counter / kmc_time
    print("Number of jumps:", counter)
    print("Jump rate:", jumprate_avg)
    relative_error = (jumprate_avg - average) / average
    assert relative_error <= 0.01, \
        "The average jump rate deviates by more than one percent!"


def test_variable_rates_index():
    """Use variable rates that are all zero except for one entry.
    Make sure that the jump always happens at that index"""
    jumprate_length = 117
    nonzero_index = 73
    jump_rates = np.zeros(jumprate_length)
    jump_rates[nonzero_index] = 0.17
    jump_rates = cycle(jump_rates)
    dt = 0.22

    fastforward_gen = MDMC.KMCLattice.fastforward_to_next_jump(jump_rates, dt)

    for i, (frame, delta_frame, kmc_time) in enumerate(fastforward_gen):
        print(frame % jumprate_length, delta_frame / jumprate_length)
        if i > 0:
            assert frame % jumprate_length == nonzero_index
        if i == 100:
            break

