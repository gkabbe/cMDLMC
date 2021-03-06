# coding=utf-8

from itertools import tee
import pathlib

import daiquiri
import numpy as np

from mdlmc.topo.topology import NeighborTopology, AngleTopology, HydroniumTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.IO.trajectory_parser import XYZTrajectory, Frame
from mdlmc.atoms.numpy_atom import dtype_xyz


logger = daiquiri.getLogger(__name__)
daiquiri.setup(daiquiri.logging.DEBUG)
daiquiri.getLogger("mdlmc.atoms.numpy_atom").setLevel(daiquiri.logging.INFO)


np.random.seed(0)


class MockTrajectory:
    def __init__(self, generator, time_step):
        self.time_step = time_step
        self.generator = generator

    def __iter__(self):
        yield from self.generator


def test_NeighborTopology_get_topology_bruteforce():
    """Assert that NeighborTopology correctly identifies the connection between the
    atoms"""
    periodic_boundaries = [10, 10, 10]
    atombox = AtomBoxCubic(periodic_boundaries)

    atoms = np.zeros((5,), dtype=dtype_xyz)
    atom_pos = np.array([[0.0, 0, 0],
                         [1.5, 0, 0],
                         [3.0, 0, 0],
                         [6.0, 0, 0],
                         [9.0, 0, 0]])

    atoms["pos"] = atom_pos
    atoms["name"] = "O"

    def trajgen():
        yield Frame.from_recarray(atoms)

    start       = [0,   0,   1,   1,   2,   4]
    destination = [1,   4,   0,   2,   1,   0]
    dist        = [1.5, 1.0, 1.5, 1.5, 1.5, 1.0]

    cutoff = 2.0

    top = NeighborTopology(MockTrajectory(trajgen(), 0.5), atombox, cutoff=cutoff, buffer=0,
                           donor_atoms="O")

    conn = top.get_topology_bruteforce(atoms["pos"])

    for st_target, de_target, di_target, st, de, di in zip(start, destination, dist, *conn):
        assert st_target == st
        assert de_target == de
        assert di_target == di


def test_NeighborTopology_get_topology_verlet_list():
    """Assert that the verlet list method yields the same results as the bruteforce
    method"""
    def trajgen():
        atoms = np.zeros((5,), dtype=dtype_xyz)
        pos = np.random.uniform(0, 10, size=(5, 3))
        atoms["name"] = "H"
        atoms["pos"] = pos
        while True:
            atoms["pos"] += np.random.normal(size=(5, 3), scale=1)
            yield Frame.from_recarray(atoms.copy())

    pbc = [10, 10, 10]
    atombox = AtomBoxCubic(pbc)

    cut, buffer = 3, 10

    traj1, traj2 = tee(trajgen())
    top1 = NeighborTopology(MockTrajectory(traj1, 0.5), atombox, cutoff=cut, buffer=buffer,
                            donor_atoms="H")
    top2 = NeighborTopology(MockTrajectory(traj2, 0.5), atombox, cutoff=cut, buffer=buffer,
                            donor_atoms="H")

    for count, (neighbors1, neighbors2) in enumerate(zip(top1.topology_verlet_list_generator(),
                                                         top2.topology_bruteforce_generator())):
        s1, d1, dist1, _ = neighbors1
        s2, d2, dist2, _ = neighbors2

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(dist1, dist2)

        if count == 50:
            break


def test_AngleTopology():
    traj_filename = pathlib.Path(__file__).absolute().parents[1] / "integration" / "trajectory.xyz"
    pbc = [29.122, 25.354, 12.363]
    atombox = AtomBoxCubic(pbc)
    xyz_trajectory = XYZTrajectory(traj_filename, time_step=0.4)
    topo = AngleTopology(xyz_trajectory, atombox, donor_atoms="O", extra_atoms="P", group_size=3,
                         cutoff=3.0, buffer=0.5)

    for x in topo:
        print(x)


def test_HydroniumTopology():
    def trajgen():
        atoms = np.zeros((8,), dtype=dtype_xyz)
        atoms["name"] = "H"
        atoms["pos"][0] = 0, 0, 0
        atoms["pos"][1] = 0, 0, 1
        atoms["pos"][2] = 0, 1, 0
        atoms["pos"][3] = 0, 1, 1
        atoms["pos"][4] = 1, 0, 0
        atoms["pos"][5] = 1, 0, 1
        atoms["pos"][6] = 1, 1, 0
        atoms["pos"][7] = 1, 1, 1

        while True:
            yield Frame.from_recarray(atoms)

    atombox = AtomBoxCubic([10, 10, 10])

    hydtop = HydroniumTopology(MockTrajectory(trajgen(), 0.5), atombox, cutoff=1.1, buffer=0,
                               donor_atoms="H")
    lattice = np.zeros(8, np.int)
    lattice[:-1] = np.arange(1, 8)
    hydtop.take_lattice_reference(lattice)


    for x in hydtop:
        print(x)
