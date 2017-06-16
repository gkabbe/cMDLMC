import unittest
import numpy as np

from mdlmc.analysis.msd import calculate_msd
from mdlmc.analysis.msd import displacement
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


class TestMSD(unittest.TestCase):
    def test_msd_normal_msd_mean(self):
        number_of_steps = 1000
        trajectory = np.zeros((number_of_steps, 1, 3))
        trajectory[:, 0, 0] = np.arange(number_of_steps)
        trajectory[:, 0, 1] = np.arange(number_of_steps)
        trajectory[:, 0, 2] = np.arange(number_of_steps)
        pbc = np.array([1000, 1000, 1000])
        intervalnumber = 100
        intervallength = 200
        msd_mean, msd_var = calculate_msd(trajectory, pbc, intervalnumber, intervallength)
        assert abs(msd_mean[100].mean() - 10000) < 0.01

    def test_msd_normal_msd_var(self):
        number_of_steps = 1000
        trajectory = np.zeros((number_of_steps, 1, 3))
        trajectory[:, 0, 0] = np.arange(number_of_steps)
        pbc = np.array([1000, 1000, 1000])
        intervalnumber = 100
        intervallength = 200
        msd_mean, msd_var = calculate_msd(trajectory, pbc, intervalnumber, intervallength)
        assert abs(msd_var.mean() - 0) < 0.01
