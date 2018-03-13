import logging

import numpy as np

from mdlmc.atoms.numpy_atom import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


logger = logging.getLogger(__name__)


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

    atoms = atoms[None, ...]

    cutoff = 2.0

    top = NeighborTopology(iter(atoms), cutoff, atombox)

    conn = next(top.get_topology_bruteforce())

    for st_target, de_target, di_target, st, de, di in zip(start, destination, dist, conn.row,
                                                           conn.col, conn.data):
        assert st_target == st
        assert de_target == de
        assert di_target == di
