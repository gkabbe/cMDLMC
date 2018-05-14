from abc import ABCMeta
import logging

import numpy as np
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d

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
        self.trajectory_time_step = trajectory.time_step
        self.cutoff = cutoff
        self.buffer = buffer
        self.atombox = atombox
        self.donor_atoms = donor_atoms

    def _determine_colvars(self, start_indices, destination_indices, distances, frame):
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
            yield self._determine_colvars(*topo)


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
        self._determine_groups()

    def _determine_groups(self):
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

    def _determine_colvars(self, start_indices, destination_indices, distances, frame):
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

    def __init__(self, trajectory: Trajectory, atombox: AtomBox, *, donor_atoms: str, cutoff: float,
                 buffer: float = 0.0,
                 distance_transformation_function: "DistanceTransformation",
                 distance_interpolator: "DistanceInterpolator") -> None:
        super().__init__(trajectory, atombox, donor_atoms=donor_atoms, cutoff=cutoff, buffer=buffer)
        # attribute will be set when calling take_lattice_reference
        # vector of length == number of protons
        # stores for each proton the time of the last jump
        self._time_of_last_jump_vec = None
        self._distance_transformation_function = distance_transformation_function
        self._distance_interpolator = distance_interpolator

    def take_lattice_reference(self, lattice):
        """Takes the lattice from class KMCLattice as a parameter and stores a reference.
        KMCLattice will check if its topology object possesses this method and will call it
        in that case."""
        self._lattice = lattice
        self._proton_number = (lattice != 0).sum()
        # save the last jump time for each proton
        # initialize with -1
        self._time_of_last_jump_vec = -np.ones(self._proton_number)
        logger.debug("Found %i protons", self._proton_number)

    def transform_distances(self, occupied_indices, distances, time):
        proton_indices = self._lattice[occupied_indices]
        last_jump_times = self._time_of_last_jump_vec[proton_indices - 1]
        residence_times = np.where(last_jump_times >= 0, time - last_jump_times, np.inf)

        if logger.isEnabledFor(logging.DEBUG):
            for k, v in locals().items():
                logger.debug("%s: %s", k, v)

        rescaled_dists = self._distance_transformation_function(distances)
        dists = self._distance_interpolator(residence_times, distances, rescaled_dists)
        return dists

    def _determine_colvars(self, start_indices, destination_indices, distances, frame):
        """"""
        n_atoms = 4
        new_start_indices = np.zeros(n_atoms * self._proton_number, int)
        new_destination_indices = np.zeros(n_atoms * self._proton_number, int)
        new_distances = np.zeros(n_atoms * self._proton_number, float)

        occupied_indices, = np.where(self._lattice)
        for i, occ_idx in enumerate(occupied_indices):
            mask = start_indices == occ_idx
            dists = distances[mask]
            destinations = destination_indices[mask]
            sorted_idx = np.argsort(dists)[:n_atoms]
            closest_indices = destinations[sorted_idx]
            closest_distances = dists[sorted_idx]

            new_start_indices[n_atoms * i: n_atoms * (i + 1)] = occ_idx
            new_destination_indices[n_atoms * i: n_atoms * (i + 1)] = closest_indices
            new_distances[n_atoms * i: n_atoms * (i + 1)] = closest_distances
            new_distances = self.transform_distances(new_start_indices, new_distances, frame.time)

        return new_start_indices, new_destination_indices, new_distances


class DistanceTransformation(metaclass=ABCMeta):
    def __call__(self, distances):
        pass


class LinearTransformation(DistanceTransformation):
    def __init__(self, a, b, d0, left_bound, right_bound):
        self.a = a
        self.b = b
        self.d0 = d0
        self.left_bound = left_bound
        self.right_bound = right_bound

    def __call__(self, distances):
        rescaled = np.where(distances < self.d0, self.b, self.a * (distances - self.d0) + self.b)
        mask = (distances <= self.left_bound) | (self.right_bound <= distances)
        rescaled[mask] = distances[mask]
        return rescaled


class InterpolatedTransformation(DistanceTransformation):
    def __init__(self, dist_array, conversion_array):
        self.interp = interp1d(dist_array, conversion_array, kind="linear")
        self.x_min, self.x_max = self.interp.x[[0, -1]]
        self.y_min = self.interp.y[0]

    def __call__(self, distances):
        inside_bounds = (self.x_min <= distances) & (distances <= self.x_max)
        rescaled = np.copy(distances)
        rescaled[inside_bounds] = self.interp(rescaled[inside_bounds])
        rescaled[rescaled < self.x_min] = self.y_min


class DistanceInterpolator:
    """Interpolates between neutral and relaxed distances"""
    def __init__(self, rescale_time):
        self.rescale_time = rescale_time

    def __call__(self, residence_time, distance_neutral, distance_relaxed):
        logger.debug("Distance neutral: %s", distance_neutral)
        logger.debug("Distance relaxed: %s", distance_relaxed)
        ratio = np.minimum(residence_time / self.rescale_time, 1)
        distance = (1 - ratio) * distance_neutral + ratio * distance_relaxed
        logger.debug("Distance interpolated: %s", distance)
        return distance
