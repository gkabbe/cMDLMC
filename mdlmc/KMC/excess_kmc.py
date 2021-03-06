# coding=utf-8

import argparse
import os
import time
import logging

import h5py
import numpy as np
from scipy.interpolate import interp1d

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5, create_dataset_from_hdf5_trajectory, \
    get_xyz_selection_from_atomname, get_hdf5_selection_from_atomname
from mdlmc.IO.config_parser import load_configfile, print_settings, print_config_template,\
    print_confighelp
from mdlmc.misc.tools import chunk
from mdlmc.cython_exts.LMC.LMCHelper import KMCRoutine, FermiFunction
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import initialize_oxygen_lattice, fastforward_to_next_jump


logger = logging.getLogger(__name__)


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) / c))


def get_hdf5_filename(trajectory_fname):
    root, ext = os.path.splitext(trajectory_fname)
    if ext in (".h5", ".hdf5"):
        return trajectory_fname
    else:
        return root + "_nobackup.hdf5"


def kmc_state_to_xyz(oxygens, protons, oxygen_lattice):
    print(oxygens.shape[0] + protons.shape[0] + 1)
    print()
    for ox in oxygens:
        print("O", " ".join([3 * "{:14.8f}"]).format(*ox))
    for p in protons:
        print("H", " ".join([3 * "{:14.8f}"]).format(*p))
    oxygen_index = np.where(oxygen_lattice > 0)[0][0]
    print("S", " ".join([3 * "{:14.8f}"]).format(*oxygens[oxygen_index]))


def trajectory_generator(trajectory, chunk_size=10000):
    """Chunks the HDF5 trajectory first before yielding it"""
    counter = 0
    while True:
        for start, stop, chk in chunk(trajectory, chunk_size=chunk_size):
            for frame in chk:
                yield counter, frame
                counter += 1


def rescale_interpolation_function(interp, dist, x_min, x_max, y_min):
    inside_bounds = np.logical_and(x_min <= dist, dist <= x_max)
    rescaled = np.copy(dist)
    rescaled[inside_bounds] = interp(rescaled[inside_bounds])
    rescaled[rescaled < x_min] = y_min
    return rescaled


def rescaled_distance_generator(distances, a, b, d0, left_bound, right_bound):
    distgen = trajectory_generator(distances)
    for counter, dist in distgen:
        rescaled = np.where(dist < d0, b, a * (dist - d0) + b)
        mask = np.logical_or(dist <= left_bound, right_bound <= dist)
        rescaled[mask] = dist[mask]
        yield counter, rescaled


def rescaled_distance_generator_interpolate(distances, dist_array, conversion_array):
    logger.info("Will interpolate conversion function from data")
    interp = interp1d(dist_array, conversion_array, kind="linear")
    x_min, x_max = interp.x[[0, -1]]
    y_min = interp.y[0]
    distgen = trajectory_generator(distances)
    logger.debug("Interpolation in range ({}, {})".format(x_min, x_max))

    for counter, dist in distgen:
        rescaled = rescale_interpolation_function(interp, dist, x_min, x_max, y_min)
        yield counter, rescaled


def last_neighbor_is_close(current_idx, last_idx, indices, dists_rescaled, dist_result,
                           check_from_old=False):
    """Checks whether a connection between the current and the last oxygen still exists (i.e.
    that either the current oxygen is in the neighbor list of the previous oxygen, or the
    previous oxygen is in the neighbor list of the current oxygen. The corresponding distance
    is then replaced by the rescaled distance."""

    logger.debug("Current idx: {}".format(current_idx))
    logger.debug("Last idx: {}".format(last_idx))
    logger.debug("Indices from current: {} -> {}".format(current_idx, indices[current_idx]))
    logger.debug("Indices from last: {} -> {}".format(last_idx, indices[last_idx]))
    logger.debug("Distances before: {}".format(dist_result))

    # Check if the last oxygen is actually still in the new oxygen's neighbor list
    idx_to_old = np.where(indices[current_idx] == last_idx)[0]
    if idx_to_old.size > 0:
        logger.debug("Connection from new to old exists")
        # If it is, the connection will be set to the rescaled distance
        dist_result[idx_to_old[0]] = dists_rescaled[current_idx, idx_to_old[0]]
        logger.debug("Distances after: {}".format(dist_result))
        return last_idx
    elif check_from_old:
        # If not, check whether the new oxygen is in the old one's neighbor list
        idx_from_old = np.where(indices[last_idx] == current_idx)[0]
        if idx_from_old.size > 0:
            logger.debug("Connection from old to new exists")
            # In that case, replace the connection with the largest distance with
            # the connection to the last neighbor
            old_neighbor_dist = dists_rescaled[last_idx, idx_from_old[0]]
            largest_dist_idx = np.argmax(dist_result)
            dist_result[largest_dist_idx] = old_neighbor_dist
            indices[current_idx, largest_dist_idx] = last_idx
            logger.debug("Distances after: {}".format(dist_result))
            logger.debug("Indices from current after: {}".format(indices[current_idx]))
            return last_idx
    # Otherwise, leave everything as it is
        else:
            logger.debug("No connection exists")

    logger.debug("Do not check connection from old")
    return last_idx


def last_neighbor_is_close_4oxys(current_idx, last_idx, indices, dists_rescaled, dist_result,
                                 check_from_old=False):
    """Finds last neighbor in indices of four neighbors, and moves it to the third place"""

    logger.debug("Using last_neighbor_is_close_4oxys")
    logger.debug("Current idx: {}".format(current_idx))
    logger.debug("Last idx: {}".format(last_idx))
    logger.debug("Indices from current: {} -> {}".format(current_idx, indices[current_idx]))
    logger.debug("Distances before: {}".format(dist_result))

    # Check if the last oxygen is actually still in the new oxygen's neighbor list
    idx_to_old = np.where(indices[current_idx] == last_idx)[0]
    if idx_to_old.size > 0:
        idx, = idx_to_old
        logger.debug("Connection from new to old exists")
        if idx == 3:
            dist_result[2] = dists_rescaled[current_idx, 3]
            indices[current_idx, 2] = indices[current_idx, 3]
        else:
            dist_result[idx] = dists_rescaled[current_idx, idx]
        logger.debug("Distances after: {}".format(dist_result))
        return last_idx
    else:
        return None


class KMCGen:
    """Class responsible for the generation of distances and jump rates.
    If relaxation_time is set, distances will be larger after a proton jump,
    and linearly decrease to the rescaled distances within the set relaxation time."""

    def __init__(self, oxy_idx, distance_gen, rescaled_distance_gen, indices,
                 jumprate_fct, jumprate_params, *, keep_last_neighbor_rescaled=False,
                 check_from_old=True, n_atoms=3):
        self._oxy_idx = oxy_idx
        self.relaxation_counter = 0
        self.relaxation_time = 0
        self.waiting_time = 0  # Don't jump while waiting time > 0
        self.distance_gen = distance_gen
        self.rescaled_distance_gen = rescaled_distance_gen
        self.indices = trajectory_generator(indices)
        self.jumprate_fct = jumprate_fct
        self.jumprate_params = jumprate_params
        # Attribute prob is set while yielding from self.jumprate_generator
        self.prob = None
        self.keep_last_neighbor_rescaled = keep_last_neighbor_rescaled
        self.check_from_old = check_from_old
        # _last_idx is set when oxy_idx's value is changed
        self._last_idx = None

        if n_atoms == 3:
            self.last_neighbor_is_close = last_neighbor_is_close
        elif n_atoms == 4:
            self.last_neighbor_is_close = last_neighbor_is_close_4oxys

    @property
    def oxy_idx(self):
        return self._oxy_idx

    @oxy_idx.setter
    def oxy_idx(self, oxy_idx):
        self._last_idx = self._oxy_idx
        self._oxy_idx = oxy_idx

    def distance_generator(self):
        distance_gen = self.distance_gen
        distance_rescaled_gen = self.rescaled_distance_gen

        keep_last_neighbor_rescaled = self.keep_last_neighbor_rescaled

        while True:
            # First, get both rescaled and unrescaled distances as well as the neighbor indices
            counter, distance_rescaled = next(distance_rescaled_gen)
            _, distance_unrescaled = next(distance_gen)
            _, indices = next(self.indices)
            logger.debug("Sweep: {}".format(counter))

            if self.relaxation_time:
                # If relaxation time is set, return a linear combination of rescaled and unrescaled
                # distances
                logger.debug("Relaxing distances:")
                if self.relaxation_counter < self.relaxation_time:
                    logger.debug("Relaxation time: {}".format(self.relaxation_counter))
                    dist = distance_unrescaled[self.oxy_idx]
                    dist_rescaled = distance_rescaled[self.oxy_idx]
                    dist_result = dist + self.relaxation_counter / self.relaxation_time * (dist_rescaled - dist)
                    self.relaxation_counter += 1
                else:
                    self.relaxation_counter = 0
                    self.relaxation_time = 0
                    dist_result = distance_rescaled[self.oxy_idx]
            else:
                # Otherwise, just get the rescaled distances
                dist_result = distance_rescaled[self.oxy_idx]

            if keep_last_neighbor_rescaled and self._last_idx is not None:
                self._last_idx = self.last_neighbor_is_close(self.oxy_idx, self._last_idx, indices,
                                                             distance_rescaled, dist_result)

            yield dist_result[:3]

    def reset_relaxationtime(self, relaxation_time):
        self.relaxation_time = relaxation_time
        self.relaxation_counter = 0

    def jumprate_generator(self):
        distance_gen = self.distance_generator()
        while True:
            for dists in distance_gen:
                if self.waiting_time > 0:
                    logger.debug("Waiting time: %s", self.waiting_time)
                    prob = np.zeros(3)
                    self.waiting_time -= 1
                else:
                    prob = self.jumprate_fct(dists, *self.jumprate_params)
                self.prob = prob
                yield prob.sum()


class PositionTracker:
    """Keeps track of the proton position and applies a distance correction
    if d_oh is specified."""
    def __init__(self, oxygen_trajectory, atombox, proton_position, d_oh=None):
        self.oxygen_trajectory = oxygen_trajectory
        if d_oh is not None:
            self.d_oh = d_oh
        else:
            self.d_oh = 0
        self.correction_vector = 0
        self.atombox = atombox
        self.proton_position = proton_position

    def get_position(self, frame_idx):
        return self.oxygen_trajectory[frame_idx, self.proton_position] + self.correction_vector

    def update_correction_vector(self, frame_idx, new_proton_position):
        # Calculate correction vector that takes into account that a proton does not really travel
        # the full O-O distance by substracting 2 * d_OH from the connection vector of the two
        # oxygen atoms between which the proton jumps
        correction_vector = self.atombox.distance(
            self.oxygen_trajectory[frame_idx, new_proton_position].astype(float),
            self.oxygen_trajectory[frame_idx, self.proton_position].astype(float))
        correction_vector /= np.sqrt(correction_vector @ correction_vector)
        correction_vector *= 2 * self.d_oh
        self.correction_vector += correction_vector
        self.proton_position = new_proton_position
        logger.debug("Correction vector: {}".format(self.correction_vector))


class Output:
    def __init__(self, oxygen_trajectory, timestep, xyz_output=False):
        self.oxygen_trajectory = oxygen_trajectory
        self.xyz_output = xyz_output
        self.output_format = "{:18d} {:18.2f} {:15.8f} {:15.8f} {:15.8f} {:10d} {:10d} {:8.2f}"
        self.timestep = timestep
        if xyz_output:
            self.print_output = self._xyzoutput
        else:
            self.print_output = self._standardoutput

    def print_columnnames(self):
        print("# {:>16} {:>18} {:>15} {:>15} {:>15} {:>10} {:>10} {:>8}".format(
            "Step", "Time", "x", "y", "z", "O-Neighbor", "Jumps", "fps"))

    def _standardoutput(self, i, proton_idx, jumps, fps):
        pos = self.oxygen_trajectory[i, proton_idx]
        print(self.output_format.format(i, i * self.timestep, *pos, proton_idx, jumps, fps),
              flush=True)

    def _xyzoutput(self, i, proton_idx, jumps, fps):
        pos = self.oxygen_trajectory[i, proton_idx]
        oxypos = self.oxygen_trajectory[i]

        print(oxypos.shape[0] + 1)
        print()
        print("H", *pos)
        for oxy in oxypos:
            print("O", *oxy)


def kmc_main(settings):
    print_settings(settings)

    # If debug is set, set global variable DEBUG = True
    if settings.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(levelname)s:%(filename)s.%(funcName)s(%(lineno)d):"
                                   " %(message)s")
    else:
        logging.basicConfig(level=logging.INFO)

    if settings.seed is not None:
        np.random.seed(settings.seed)

    trajectory_fname = settings.filename
    verbose = settings.verbose
    chunk_size = settings.chunk_size
    overwrite_oxygen_trajectory = settings.overwrite_oxygen_trajectory
    trajectory_by_mdconvert = settings.mdconvert_trajectory

    hdf5_fname = get_hdf5_filename(trajectory_fname)
    logger.info("Looking for HDF5 file {}".format(hdf5_fname))
    if not os.path.exists(hdf5_fname):
        logger.info("Could not find HDF5 file. Will create it now.")
        oxygen_selection = get_xyz_selection_from_atomname(trajectory_fname, "O")
        save_trajectory_to_hdf5(trajectory_fname, hdf5_fname, remove_com_movement=True,
                                verbose=verbose, dataset_name="oxygen_trajectory",
                                selection=oxygen_selection)

    hdf5_file = h5py.File(hdf5_fname, "a")
    try:
        oxygen_trajectory = hdf5_file["oxygen_trajectory"]
    except KeyError:
        logger.info("Could not find dataset oxygen_trajectory in {}".format(hdf5_fname))
        logger.info("Assuming that this hdf5 file was created by mdconvert")
        logger.info("Try to load trajectory")
        overwrite_oxygen_trajectory = True
        trajectory_by_mdconvert = True

    selection, _ = get_hdf5_selection_from_atomname(hdf5_fname, "O")
    # Create selection function which selects the oxygen atom coordinates
    # and converts them from nm to angstrom
    if trajectory_by_mdconvert:
        # If it is an mdconvert trajectory, units need to be converted from nm to angstrom
        def selection_fct(arr):
            return 10 * np.take(arr, selection, axis=1)
    else:
        def selection_fct(arr):
            return np.take(arr, selection, axis=1)

    oxygen_trajectory = create_dataset_from_hdf5_trajectory(hdf5_file, hdf5_file["coordinates"],
                                                            "oxygen_trajectory",
                                                            selection_fct,
                                                            overwrite=overwrite_oxygen_trajectory,
                                                            chunk_size=1000)

    trajectory_length = oxygen_trajectory.shape[0]
    timestep_md = settings.md_timestep_fs
    total_sweeps = settings.sweeps
    xyz_output = settings.xyz_output
    print_frequency = settings.print_frequency
    relaxation_time = settings.relaxation_time
    waiting_time = settings.waiting_time
    d_oh = settings.d_oh

    oxygen_number = oxygen_trajectory.shape[1]

    if settings.sweeps > trajectory_length:
        raise ValueError("Number of sweeps are {}, but oxygen trajectory only"
                " has length {}!".format(settings.sweeps, trajectory_length))
    # Initialize with one excess proton
    if settings.start_position is not None:
        proton_position = settings.start_position
        oxygen_lattice = np.zeros(oxygen_number, np.uint8)
        oxygen_lattice[proton_position] = 1
    else:
        oxygen_lattice = initialize_oxygen_lattice(oxygen_number, 1)
        proton_position, = np.where(oxygen_lattice)[0]

    atombox_cubic = AtomBoxCubic(settings.pbc)

    pos_tracker = PositionTracker(oxygen_trajectory, atombox_cubic, proton_position, d_oh)
    output = Output(oxygen_trajectory, timestep_md, settings.xyz_output)

    if settings.seed is not None:
        np.random.seed(settings.seed)

    a, b, c = (settings.jumprate_params_fs["a"], settings.jumprate_params_fs["b"],
               settings.jumprate_params_fs["c"])
    # Multiply with timestep, to get the jump probability within one timestep
    a *= timestep_md

    jumprate_function = FermiFunction(a, b, c)
    kmc = KMCRoutine(atombox_cubic, oxygen_lattice, jumprate_function, n_atoms=settings.n_atoms)

    logger.info("Creating array of distances")
    distances_name = "distances_{}".format(settings.n_atoms)
    indices_name = "indices_{}".format(settings.n_atoms)
    distances, indices = create_dataset_from_hdf5_trajectory(hdf5_file, oxygen_trajectory,
                                                             (distances_name, indices_name),
                                                             kmc.determine_distances,
                                                             chunk_size, dtype=(np.float32, np.int32),
                                                             overwrite=settings.overwrite_jumprates)

    if not xyz_output:
        output.print_columnnames()

    distance_gen = trajectory_generator(distances)
    if settings.no_rescaling:
        rescaled_distance_gen = trajectory_generator(distances)
    elif settings.conversion_data:
        data = np.loadtxt(settings.conversion_data)
        dist, *_, conversion = data.T
        rescaled_distance_gen = rescaled_distance_generator_interpolate(distances, dist, conversion)
        del dist
        del conversion
    else:
        rescaled_distance_gen = rescaled_distance_generator(distances,
                                                            **settings.rescale_parameters)

    kmc_gen = KMCGen(proton_position, distance_gen, rescaled_distance_gen, indices, fermi,
                     (a, b, c), keep_last_neighbor_rescaled=settings.keep_last_neighbor_rescaled,
                     check_from_old=settings.check_from_old, n_atoms=settings.n_atoms)
    fastforward_gen = fastforward_to_next_jump(kmc_gen.jumprate_generator(), timestep_md)

    kmc_time, frame, sweep, jumps = 0, 0, 0, 0

    start_time = time.time()
    while sweep < total_sweeps:
        next_sweep, delta_frame, kmc_time = next(fastforward_gen)
        frame = sweep % trajectory_length

        for i in range(sweep, next_sweep):
            if i % print_frequency == 0:
                output.print_output(i, proton_position, jumps, i / (time.time() - start_time))

        jumps += 1
        sweep = next_sweep
        logger.debug("Sweep {}".format(sweep))
        logger.debug("Jumping")
        logger.debug("Old proton position: {}".format(proton_position))

        probs = kmc_gen.prob
        cumsum = np.cumsum(probs)
        neighbor_indices = indices[frame, proton_position]

        logger.debug("Choose between {}".format(neighbor_indices))
        logger.debug("With probabilities {}".format(probs))

        random_draw = np.random.uniform(0, cumsum[-1])
        ix = np.searchsorted(cumsum, random_draw)
        proton_position = neighbor_indices[ix]
        kmc_gen.oxy_idx = proton_position
        # After a jump, the relaxation time is increased
        kmc_gen.reset_relaxationtime(relaxation_time)
        kmc_gen.waiting_time = waiting_time

        logger.debug("New proton position: {}".format(proton_position))

        if d_oh:
            pos_tracker.update_correction_vector(sweep % trajectory_length, proton_position)


def main(*args):
    parser = argparse.ArgumentParser(description="KMC",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_load = subparsers.add_parser("load", help="Load config file")
    parser_config_help = subparsers.add_parser("config_help", help="Config file help")
    parser_config_file = subparsers.add_parser("config_file", help="Print config file template")
    parser_config_file.add_argument("--sorted", "-s", action="store_true",
                                    help="Sort config parameters lexicographically")
    parser_load.add_argument("config_file", help="Config file")
    parser.set_defaults(func=kmc_main)
    args = parser.parse_args()
    if args.subparser_name == "config_file":
        print_config_template("KMCWater", args.sorted)
    elif args.subparser_name == "config_help":
        print_confighelp("KMCWater")
    else:
        settings = load_configfile(args.config_file, config_name="KMCWater")
        args.func(settings)
