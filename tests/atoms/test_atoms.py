import daiquiri
import numpy as np

from mdlmc.atoms.numpy_atom import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


logger = daiquiri.getLogger(__name__)
daiquiri.setup(daiquiri.logging.DEBUG)


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

    top = NeighborTopology(iter(atoms), cutoff, atombox)

    conn = top.get_topology_bruteforce(atoms)

    for st_target, de_target, di_target, st, de, di in zip(start, destination, dist, *conn):
        assert st_target == st
        assert de_target == de
        assert di_target == di


def test_NeighborTopology_get_topology_verlet_list():
    def trajgen():
        atoms = np.random.uniform(0, 10, size=(5, 3))
        while True:
            atoms += np.random.normal(size=(5, 3), scale=4)
            yield atoms.copy()

    pbc = [10, 10, 10]
    atombox = AtomBoxCubic(pbc)

    top = NeighborTopology(trajgen(), 3.0, atombox)

    for i, (start, dest, dist) in enumerate(top.get_topology_verlet_list()):
        logger.info("%s %s %s", start, dest, dist)
        if i == 5:
            break

