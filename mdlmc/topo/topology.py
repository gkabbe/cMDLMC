import logging
import numpy as np
from scipy.sparse import lil_matrix

from ..IO.trajectory_parser import Trajectory
from ..misc.tools import cache_last_elements
from mdlmc.cython_exts.LMC.PBCHelper import AtomBox


logger = logging.getLogger(__name__)


class NeighborTopology:
    """Keeps track of the connections between donor/acceptor atoms.
    Given a cutoff distance, for each atom the atoms within this
    distance will be determined."""
    def __init__(self, trajectory: Trajectory, atombox: AtomBox, *, donor_atoms: str, cutoff: float,
                 buffer: float = 0.0) -> None:
        """

        Parameters
        ----------
        trajectory: Trajectory
            Trajectory object from which the atomic positions are obtained
        atombox: AtomBox
            AtomBox object
        donor_atoms: str
            Name of the atom type of the donor / acceptor atoms
        cutoff: float
            Cutoff region for pairwise calculations
        buffer: float
            Buffer region which is considered before recalculating the topology of closest atoms
        """

        self.trajectory, self.get_cached_frames = cache_last_elements(trajectory)
        self.cutoff = cutoff
        self.buffer = buffer
        self.atombox = atombox
        self.donor_atoms = donor_atoms

    def determine_colvars(self, start_indices, destination_indices, distances, frame):
        """Method for the determination of all necessary collective variables.
        Per convention, the first collective variable should always be the distance."""
        return start_indices, destination_indices, distances

    def get_topology_bruteforce(self, frame):
        """Determine the distance for each atom pair.
        If it is below cutoff + buffer, add it to the list
        of connections."""

        logger.debug("Determine topology")
        topology_matrix = lil_matrix((frame.shape[0], frame.shape[0]), dtype=float)
        for i, atom1 in enumerate(frame):
            for j in range(i):
                atom2 = frame[j]
                if i != j:
                    dist = self.atombox.length(atom1, atom2)
                    if dist <= self.cutoff + self.buffer:
                        topology_matrix[i, j] = dist
                        topology_matrix[j, i] = dist
        connections = topology_matrix.tocoo()
        logger.debug("Topology matrix: %s", connections)
        return connections.row, connections.col, connections.data

    def topology_bruteforce_generator(self):
        for full_frame in self.trajectory:
            frame = full_frame[self.donor_atoms].atom_positions
            topo = self.get_topology_bruteforce(frame)
            yield (*topo, full_frame)

    def topology_verlet_list_generator(self):
        """Keep track of the two maximum atom displacements.
        As soon as their sum is larger than the buffer region, update
        the neighbor topology."""

        last_frame = None
        atombox = self.atombox
        logger.debug("Start Verlet list")
        displacement = 0

        for full_frame in self.trajectory:
            frame = full_frame[self.donor_atoms].atom_positions
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

            yield (*topology, full_frame)
            last_frame = frame

    def __iter__(self):
        for topo in self.topology_verlet_list_generator():
            yield self.determine_colvars(*topo)


class AngleTopology(NeighborTopology):
    """This topology class is used to calculate the POO angle as an additional collective variable.
    Of course, other atom types are possible as well. In that case, the parameters for donor_atoms
    and extra_atoms just need to be changed accordingly.

                             O --- O
                            /
                           P

    """
    def __init__(self, trajectory: Trajectory, atombox: AtomBox, *, donor_atoms: str,
                 extra_atoms: str, group_size: int, cutoff: float, buffer: float = 0.0) -> None:
        super().__init__(trajectory, atombox, donor_atoms=donor_atoms, cutoff=cutoff, buffer=buffer)

        self.extra_atoms = extra_atoms
        self.group_size = group_size
        self.determine_groups()

    def determine_groups(self):
        """Find for each phosphorus atom the three closest oxygen atoms. This way, the donor atoms
        belonging to one phosphonic group can be found."""
        first_frame = next(iter(self.trajectory))
        distances_PO = self.atombox.length_all_to_all(first_frame[self.extra_atoms].atom_positions,
                                                      first_frame[self.donor_atoms].atom_positions)

        closest_Os = np.argsort(distances_PO, axis=1)[:, :self.group_size]
        logger.debug("Groups of donor atoms:\n%s", closest_Os)

        self.map_O_to_P = {}
        for P_index, Os in enumerate(closest_Os):
            for O_index in Os:
                self.map_O_to_P[O_index] = P_index
        logger.debug("Mapping:\n%s", self.map_O_to_P)

    def determine_colvars(self, start_indices, destination_indices, distances, frame):
        """Determine here the POO angles"""
        angles = np.empty(start_indices.shape, dtype=float)
        p_atoms = frame[self.extra_atoms].atom_positions
        o_atoms = frame[self.donor_atoms].atom_positions
        for i, (O_i, O_j) in enumerate(zip(start_indices, destination_indices)):
            P_i = self.map_O_to_P[O_i]
            angles[i] = self.atombox.angle(p_atoms[P_i], o_atoms[O_i], o_atoms[O_j])

        return (start_indices, destination_indices, distances, angles)


class HydroniumTopology(NeighborTopology):
    """Mimics the neighbor topology of a H3O+ ion in water by only defining connections to the
    three closest oxygen neighbors."""

    def take_lattice_reference(self, lattice):
        """Takes the lattice from class KMCLattice as a parameter and stores a reference.
        KMCLattice will check if its topology object possesses this method and will call it
        in that case."""
        self._lattice = lattice

    def determine_colvars(self, start_indices, destination_indices, distances, frame):
        """"""
        hyd_idx, = np.where(self._lattice)
        valid_idx = np.in1d(start_indices, hyd_idx)
        for i in hyd_idx:
            start_indices == i

        list_of_closest_four = np.array(
            [np.argsort(distances[start_indices == i])[:4] for i in hyd_idx])

        import ipdb; ipdb.set_trace()
        return start_indices[hyd_idx], destination_indices[hyd_idx], distances[hyd_idx]
