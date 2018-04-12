from itertools import tee
import daiquiri
import numpy as np
import pytest

from mdlmc.topo.topology import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


logger = daiquiri.getLogger(__name__)
daiquiri.setup(daiquiri.logging.DEBUG)
daiquiri.getLogger("mdlmc.atoms.numpy_atom").setLevel(daiquiri.logging.INFO)


np.random.seed(0)


def test_NeighborTopology_get_topology_bruteforce():
    """Assert that NeighborTopology correctly identifies the connection between the
    atoms"""
    periodic_boundaries = [10, 10, 10]
    atombox = AtomBoxCubic(periodic_boundaries)

    atoms = np.array([[0.0, 0, 0],
                      [1.5, 0, 0],
                      [3.0, 0, 0],
                      [6.0, 0, 0],
                      [9.0, 0, 0]])

    start       = [0,   0,   1,   1,   2,   4]
    destination = [1,   4,   0,   2,   1,   0]
    dist        = [1.5, 1.0, 1.5, 1.5, 1.5, 1.0]

    cutoff = 2.0

    top = NeighborTopology(iter(atoms), atombox, cutoff=cutoff, buffer=0, donor_atoms=slice(None))

    conn = top.get_topology_bruteforce(atoms)

    for st_target, de_target, di_target, st, de, di in zip(start, destination, dist, *conn):
        assert st_target == st
        assert de_target == de
        assert di_target == di


def test_NeighborTopology_get_topology_verlet_list():
    """Assert that the verlet list method yields the same results as the bruteforce
    method"""
    def trajgen():
        atoms = np.random.uniform(0, 10, size=(5, 3))
        while True:
            atoms += np.random.normal(size=(5, 3), scale=1)
            yield atoms.copy()

    pbc = [10, 10, 10]
    atombox = AtomBoxCubic(pbc)

    cut, buffer = 3, 10

    traj1, traj2 = tee(trajgen())
    top1 = NeighborTopology(traj1, atombox, cutoff=cut, buffer=buffer, donor_atoms=slice(None))
    top2 = NeighborTopology(traj2, atombox, cutoff=cut, buffer=buffer, donor_atoms=slice(None))

    for count, (neighbors1, neighbors2) in enumerate(zip(top1.topology_verlet_list_generator(),
                                                         top2.topology_bruteforce_generator())):
        s1, d1, dist1, _ = neighbors1
        s2, d2, dist2, _ = neighbors2

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(dist1, dist2)

        if count == 50:
            break
