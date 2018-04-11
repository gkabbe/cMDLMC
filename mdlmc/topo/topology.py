import numpy as np
from scipy.sparse import lil_matrix

from mdlmc.atoms.numpy_atom import logger
from mdlmc.cython_exts.LMC.PBCHelper import AtomBox


class NeighborTopology:
    """Keeps track of the connections between donor/acceptor atoms.
    Given a cutoff distance, for each atom the atoms within this
    distance will be determined."""
    def __init__(self, trajectory, atombox: AtomBox, *, donor_atoms: str, cutoff: float,
                 buffer: float = 0.0) -> None:
        """

        Parameters
        ----------
        trajectory
        atombox
        donor_atoms
        cutoff
        buffer
        """

        self.trajectory = trajectory
        self.cutoff = cutoff
        self.buffer = buffer
        self.atombox = atombox
        self.donor_atoms = donor_atoms.encode()

    def _get_selection(self, trajectory):
        """Given a trajectory array with fields name and pos,
        yield the array with the atom positions.
        """
        for frame in trajectory:
            yield frame["pos"][frame["name"] == self.donor_atoms]

    def get_topology_bruteforce(self, frame):
        """Determine the distance for each atom pair.
        If it is below cutoff + buffer, add it to the list
        of connections."""

        logger.debug("Determine topology")
        topology_matrix = lil_matrix((frame.shape[0], frame.shape[0]), dtype=float)
        for i, atom1 in enumerate(frame):
            for j in range(i):
                atom2 =  frame[j]
                if i != j:
                    dist = self.atombox.length(atom1, atom2)
                    if dist <= self.cutoff + self.buffer:
                        topology_matrix[i, j] = dist
                        topology_matrix[j, i] = dist
        connections = topology_matrix.tocoo()
        logger.debug("Topology matrix: %s", connections)
        return connections.row, connections.col, connections.data

    def topology_bruteforce_generator(self):
        for frame in self._get_selection(self.trajectory):
            topo = self.get_topology_bruteforce(frame)
            yield topo

    def topology_verlet_list_generator(self):
        """Keep track of the two maximum atom displacements.
        As soon as their sum is larger than the buffer region, update
        the neighbor topology."""

        last_frame = None
        atombox = self.atombox
        logger.debug("start verlet list")
        displacement = 0

        for frame in self._get_selection(self.trajectory):
            if last_frame is None:
                logger.debug("First frame. Get topo by bruteforce")
                topology = self.get_topology_bruteforce(frame)
                logger.debug(topology)
                dr = np.zeros(frame.shape[0])
            else:
                dr = atombox.length(last_frame, frame)

            displacement += dr
            displ_max1, displ_max2 = np.sort(displacement)[-2:]
            logger.debug("displ_max1 = %s; displ_max2 = %s", displ_max1, displ_max2)

            if displ_max1 + displ_max2 > self.buffer:
                logger.debug("Buffer region crossed. Recalculating topology")
                topology = self.get_topology_bruteforce(frame)
                displacement = 0
            else:
                logger.debug("Topology has not changed")
                dist = atombox.length(frame[topology[0]], frame[topology[1]])
                topology = (*topology[:2], dist)

            yield topology
            last_frame = frame


class NeighborTopArray(NeighborTopology):
    """If the trajectory generator directly yields Numpy arrays,
    use this class"""
    def _get_selection(self, trajectory):
        for frame in trajectory:
            yield frame