# coding=utf-8

import numpy as np
import pytest

from mdlmc.LMC.output import MeanSquareDisplacement
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


@pytest.fixture
def atom_positions():
    pos = np.arange(1, 19).reshape(6, 3)
    return pos


@pytest.fixture
def lattice():
    return np.array([0, 3, 0, 0, 1, 2])


def test_MeanSquareDisplacement_position(atom_positions, lattice):
    atombox = AtomBoxCubic([10, 10, 10])
    msd = MeanSquareDisplacement(atom_positions, lattice, atombox=atombox)
    output_desired = np.array([[13, 14, 15],
                               [16, 17, 18],
                               [4, 5, 6]])

    np.testing.assert_equal(msd.snapshot, output_desired)


def test_MeanSquareDisplacement_displacement(atom_positions, lattice):
    atombox = AtomBoxCubic([10, 10, 10])
    msd = MeanSquareDisplacement(atom_positions, lattice, atombox=atombox)
    # protons 1 and 2 swap positions
    lattice[-2], lattice[-1] = lattice[-1], lattice[-2]
    msd.update_displacement(atom_positions, lattice)
    displacement = np.zeros((3, 3), int)
    displacement[0] = [3, 3, 3]
    displacement[1] = [-3, -3, -3]

    np.testing.assert_equal(msd.displacement, displacement)

    # proton 2 jumps to empty site
    lattice[-2], lattice[-3] = lattice[-3], lattice[-2]
    msd.update_displacement(atom_positions, lattice)
    displacement[1] += np.array([-3, -3, -3])
    np.testing.assert_equal(msd.displacement, displacement)
