# coding=utf-8

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
    __show_in_config__ = True
    __no_config_parameter__ = ["trajectory", "atom_box"]

    def __init__(self, trajectory: Trajectory, atom_box: AtomBox, *, donor_atoms: str,
                 cutoff: float = 3.0, buffer: float = 2.0) -> None:
        """

        Parameters
        ----------
        trajectory: Trajectory
            Trajectory object from which the atomic positions are obtained
        atom_box: AtomBox
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
        self.atombox = atom_box
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

    def update_time_of_last_jump(self, proton_idx, new_time):
        pass


class AngleTopology(NeighborTopology):
    """This topology class is used to calculate the POO angle as an additional collective variable.
    Of course, other atom types are possible as well. In that case, the parameters for donor_atoms
    and extra_atoms just need to be changed accordingly.

                             O -- O
                            /
                           P

    """
    def __init__(self, trajectory: Trajectory, atom_box: AtomBox, *, donor_atoms: str,
                 extra_atoms: str, group_size: int, cutoff: float = 3.0, buffer: float = 2.0) -> None:
        super().__init__(trajectory, atom_box, donor_atoms=donor_atoms, cutoff=cutoff, buffer=buffer)

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

    __no_config_parameter__ = ["trajectory", "atom_box", "distance_transformation_function",
                               "distance_interpolator"]

    def __init__(self, trajectory: Trajectory, atom_box: AtomBox, *, donor_atoms: str, cutoff: float,
                 buffer: float = 0.0,
                 distance_transformation_function: "DistanceTransformation" = None,
                 distance_interpolator: "DistanceInterpolator" = None
                 ) -> None:
        super().__init__(trajectory, atom_box, donor_atoms=donor_atoms, cutoff=cutoff, buffer=buffer)
        # attribute will be set when calling take_lattice_reference
        # vector of length == number of protons
        # stores for each proton the time of the last jump
        self._time_of_last_jump_vec = None
        if distance_transformation_function:
            self._distance_transformation_function = distance_transformation_function
        else:
            logger.info("Distance transformation function not specified."
                        "Will not rescale distances.")
            self._distance_transformation_function = lambda distance: distance

        if distance_interpolator:
            self._distance_interpolator = distance_interpolator
        else:
            logger.info("Distance interpolator not specified."
                        "Will rescale distances without delay.")
            self._distance_interpolator = lambda time, d_neutral, d_relaxed: d_relaxed

    def take_lattice_reference(self, lattice):
        """Takes the lattice from class KMCLattice as a parameter and stores a reference.
        KMCLattice will check if its topology object possesses this method and will call it
        in that case."""
        self._lattice = lattice.view()
        # Make array view read-only
        self._lattice.flags.writeable = False
        self._proton_number = (lattice != 0).sum()
        # save the last jump time for each proton
        # initialize with -1
        self._time_of_last_jump_vec = -np.ones(self._proton_number)
        logger.debug("Found %i protons", self._proton_number)

    def transform_distances(self, occupied_indices, distances, time):
        occupied_indices = np.unique(occupied_indices)
        logger.debug("Occupied indices: %s", occupied_indices)
        proton_indices = self._lattice[occupied_indices]
        last_jump_times = self._time_of_last_jump_vec[proton_indices - 1]
        residence_times = np.where(last_jump_times >= 0, time - last_jump_times, np.inf)

        if logger.isEnabledFor(logging.DEBUG):
            for k, v in locals().items():
                logger.debug("%s: %s", k, v)

        rescaled_dists = self._distance_transformation_function(distances)
        try:
            dists = self._distance_interpolator(residence_times, distances, rescaled_dists)
        except:
            import ipdb; ipdb.set_trace()
        return dists

    def _determine_colvars(self, start_indices, destination_indices, distances, frame):
        """"""
        n_atoms = 4
        donor_nr = len(self._lattice)
        new_start_indices = np.zeros((donor_nr, n_atoms), int)
        new_destination_indices = np.zeros((donor_nr, n_atoms), int)
        new_distances = np.zeros((donor_nr, n_atoms), float)

        # get the donor indices which are occupied
        for occ_idx in range(donor_nr):
            mask = start_indices == occ_idx
            dists = distances[mask]
            destinations = destination_indices[mask]
            sorted_idx = np.argsort(dists)[:n_atoms]
            closest_indices = destinations[sorted_idx]
            closest_distances = dists[sorted_idx]

            new_start_indices[occ_idx] = occ_idx
            new_destination_indices[occ_idx] = closest_indices
            new_distances[occ_idx] = closest_distances
        new_distances = self.transform_distances(new_start_indices, new_distances, frame.time)

        return new_start_indices.flatten(), new_destination_indices.flatten(), new_distances.flatten()

    def update_time_of_last_jump(self, proton_idx, new_time):
        self._time_of_last_jump_vec[proton_idx - 1] = new_time


class DistanceTransformation(metaclass=ABCMeta):
    """If a topology which supports a transformation of the donor-acceptor distances is chosen
    (for example HydroniumTopology), this class specifies how the donor-acceptor distances
    are rescaled."""

    __show_in_config__ = True

    def __call__(self, distances):
        pass


class ReLUTransformation(DistanceTransformation):
    """Rectified Linear Unit transformation.
    Returns a constant value b for distances below d0.
    Returns linear function for distances above d0"""

    def __init__(self,
                 a: float,
                 b: float,
                 d0: float,
                 left_bound: float,
                 right_bound: float) -> None:
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
    """Transform O-O distances using the Scipy interp1d routine."""

    __show_signature_of__ = "from_file"

    def __init__(self, dist_array, conversion_array):
        self.interp = interp1d(dist_array, conversion_array, kind="linear")
        self.x_min, self.x_max = self.interp.x[[0, -1]]
        self.y_min = self.interp.y[0]

    @classmethod
    def from_file(cls,
                  dist_array_filename: str,
                  conversion_array_filename: str
                  ) -> "InterpolatedTransformation":
        """

        Parameters
        ----------
        dist_array_filename:
            filename of array containing the unrescaled distances
        conversion_array_filename:
            filename of array containing the rescaled distance values

        Returns
        -------
        InterpolatedTransformation object
        """

        dist_array = np.load(dist_array_filename)
        conversion_array = np.load(conversion_array_filename)

        return cls(dist_array, conversion_array)

    def __call__(self, distances):
        inside_bounds = (self.x_min <= distances) & (distances <= self.x_max)
        rescaled = np.copy(distances)
        rescaled[inside_bounds] = self.interp(rescaled[inside_bounds])
        rescaled[rescaled < self.x_min] = self.y_min
        return rescaled


class DistanceInterpolator:
    """Interpolates between neutral and relaxed distances.
    Rescales linearly in time between neutral and rescaled donor-acceptor distances.
    Only useful in combination with HydroniumTopology (or any other distance-rescaling topology class)"""

    __show_in_config__ = True

    def __init__(self, relaxation_time: float):
        self.relaxation_time = relaxation_time

    def __call__(self, residence_time, distance_neutral, distance_relaxed):
        logger.debug("Distance neutral: %s", distance_neutral)
        logger.debug("Distance relaxed: %s", distance_relaxed)
        ratio = np.minimum(residence_time / self.relaxation_time, 1)[:, None]
        distance = (1 - ratio) * distance_neutral + ratio * distance_relaxed
        logger.debug("Distance interpolated: %s", distance)
        return distance
