import argparse
import os
import time

import tables
import h5py
import numpy as np

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5, create_dataset_from_hdf5_trajectory, \
    get_selection_from_atomname
from mdlmc.IO.config_parser import load_configfile, print_settings, print_config_template,\
    print_confighelp
from mdlmc.misc.tools import chunk
from mdlmc.cython_exts.LMC.LMCHelper import KMCRoutine, FermiFunction
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxWaterLinearConversion, \
    AtomBoxWaterRampConversion
from mdlmc.LMC.MDMC import initialize_oxygen_lattice


DEBUG = False


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) / c))


def get_hdf5_filename(trajectory_fname):
    root, ext = os.path.splitext(trajectory_fname)
    if ext == ".hdf5":
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


def fastforward_to_next_jump(jumprates, dt):
    """Implements Kinetic Monte Carlo with time-dependent rates.

    Parameters
    ----------
    jumprates : generator / iterator
        Unit: femtosecond^{-1}
        Proton jump rate from an oxygen site to any neighbor
    proton_position : int
        Index of oxygen at which the excess proton is residing.
    dt : float
        Trajectory time step
    frame : int
        Start frame
    time : float
        Start time

    Returns
    -------
    frame: int
        Frame at which the next event occurs
    delta_frame : int
        Difference between frame and the index at which the next event occurs
    delta_t : float
        Difference between current time and the time of the next event
    """

    sweep, kmc_time = 0, 0

    current_rate = next(jumprates)
    while True:
        time_selector = -np.log(1 - np.random.random())

        # Handle case where time selector is so small that the next frame is not reached
        t_trial = time_selector / current_rate
        if (kmc_time + t_trial) // dt == kmc_time // dt:
            kmc_time += t_trial
            delta_frame = 0
        else:
            delta_t, delta_frame = dt - kmc_time % dt, 1
            current_probsum = current_rate * delta_t
            next_rate = next(jumprates)
            next_probsum = current_probsum + next_rate * dt

            while next_probsum < time_selector:
                delta_frame += 1
                current_probsum = next_probsum
                next_rate = next(jumprates)
                next_probsum = current_probsum + next_rate * dt

            rest = time_selector - current_probsum
            delta_t += (delta_frame - 1) * dt + rest / next_rate
            kmc_time += delta_t
        sweep += delta_frame
        yield sweep, delta_frame, kmc_time


def trajectory_generator(trajectory, chunk_size=10000):
    """Chunks the HDF5 trajectory first before yielding it"""
    counter = 0
    while True:
        for start, stop, chk in chunk(trajectory, chunk_size=chunk_size):
            for frame in chk:
                yield counter, frame
                counter += 1


class KMCGen:
    """Class responsible for the generation of distances and jump rates.
    If relaxation_time is set, distances will be larger after a proton jump,
    and linearly decrease to the rescaled distances within the set relaxation time."""

    def __init__(self, oxy_idx, distances, distances_rescaled, jumprate_fct, jumprate_params):
        self.oxy_idx = oxy_idx
        self.relaxation_time = 0
        self.waiting_time = 0  # Don't jump while waiting time > 0
        self.distances = distances
        self.distances_rescaled = distances_rescaled
        self.jumprate_fct = jumprate_fct
        self.jumprate_params = jumprate_params
        # Attribute prob is set while yielding from self.jumprate_generator
        self.prob = None

    def distance_generator(self):
        distance_gen = trajectory_generator(self.distances)
        distance_rescaled_gen = trajectory_generator(self.distances_rescaled)

        while True:
            if self.relaxation_time:
                if DEBUG:
                    print("Relaxing distances:")
                for i in range(self.relaxation_time):
                    _, dist = next(distance_gen)
                    dist = dist[self.oxy_idx]
                    counter, dist_rescaled = next(distance_rescaled_gen)
                    dist_rescaled = dist_rescaled[self.oxy_idx]
                    yield dist + i / self.relaxation_time * (dist_rescaled - dist)
                self.relaxation_time = 0
            else:
                counter, dist = next(distance_rescaled_gen)
                dist = dist[self.oxy_idx]
                yield dist

    def jumprate_generator(self):
        distance_gen = self.distance_generator()
        while True:
            for dists in distance_gen:
                if self.waiting_time > 0:
                    prob = np.zeros(3)
                    self.waiting_time -= 1
                else:
                    prob = self.jumprate_fct(dists, *self.jumprate_params)
                self.prob = prob
                yield prob.sum()


def kmc_main(settings):
    print_settings(settings)

    # If debug is set, set global variable DEBUG = True
    if settings.debug:
        global DEBUG
        DEBUG = True

    trajectory_fname = settings.filename
    verbose = settings.verbose
    chunk_size = settings.chunk_size

    hdf5_fname = get_hdf5_filename(trajectory_fname)
    if verbose:
        print("# Looking for HDF5 file", hdf5_fname)
    if not os.path.exists(hdf5_fname):
        if verbose:
            print("# Could not find HDF5 file. Will create it now.")
        oxygen_selection = get_selection_from_atomname(trajectory_fname, "O")
        save_trajectory_to_hdf5(trajectory_fname, hdf5_fname, remove_com_movement=True,
                                verbose=verbose, dataset_name="oxygen_trajectory",
                                selection=oxygen_selection)

    hdf5_file = h5py.File(hdf5_fname, "a")
    oxygen_trajectory = hdf5_file["oxygen_trajectory"]

    trajectory_length = oxygen_trajectory.shape[0]
    timestep_md = settings.md_timestep_fs
    total_sweeps = settings.sweeps
    xyz_output = settings.xyz_output
    print_frequency = settings.print_frequency
    relaxation_time = settings.relaxation_time

    oxygen_number = oxygen_trajectory.shape[1]
    # Initialize with one excess proton
    oxygen_lattice = initialize_oxygen_lattice(oxygen_number, 1)
    proton_position, = np.where(oxygen_lattice)[0]

    atombox_cubic = AtomBoxCubic(settings.pbc)

    if settings.seed is not None:
        np.random.seed(settings.seed)

    if settings.rescale_parameters:
        if settings.rescale_function == "linear":
            atombox_rescale = AtomBoxWaterLinearConversion(settings.pbc,
                                                           settings.rescale_parameters)
        elif settings.rescale_function == "ramp_function":
            atombox_rescale = AtomBoxWaterRampConversion(settings.pbc, settings.rescale_parameters)
        else:
            raise ValueError("Unknown rescale function name", settings.rescale_function)

    a, b, c = (settings.jumprate_params_fs["a"], settings.jumprate_params_fs["b"],
               settings.jumprate_params_fs["c"])

    jumprate_function = FermiFunction(a, b, c)
    kmc = KMCRoutine(atombox_cubic, oxygen_lattice, jumprate_function)
    if settings.rescale_parameters:
        kmc_rescale = KMCRoutine(atombox_rescale, oxygen_lattice, jumprate_function)

    print("# Creating array of distances", file=settings.output)
    distances, indices = create_dataset_from_hdf5_trajectory(hdf5_file, oxygen_trajectory,
                                                             ("distances", "indices"),
                                                             kmc.determine_distances,
                                                             chunk_size, dtype=(np.float32, np.int32),
                                                             overwrite=settings.overwrite_jumprates)

    if settings.rescale_parameters:
        if verbose:
            print("# Creating array of rescaled distances")
        distances_rescaled, indices = create_dataset_from_hdf5_trajectory(
            hdf5_file, oxygen_trajectory, ("distances_rescaled", "indices"),
            kmc_rescale.determine_distances, chunk_size, dtype=(np.float32, np.int32),
            overwrite=settings.overwrite_jumprates)

    if settings.no_rescaling or not settings.rescale_parameters:
        distances_rescaled = distances

    output_format = "{:18d} {:18.2f} {:15.8f} {:15.8f} {:15.8f} {:10d} {:8.2f} fps"

    kmc_gen = KMCGen(proton_position, distances, distances_rescaled, fermi, (a, b, c))
    fastforward_gen = fastforward_to_next_jump(kmc_gen.jumprate_generator(), timestep_md)

    if DEBUG:
        distance_debug = trajectory_generator(distances)
        distance_rescaled_debug = trajectory_generator(distances_rescaled)

    kmc_time, frame, sweep, jumps = 0, 0, 0, 0

    start_time = time.time()
    while sweep < total_sweeps:
        next_sweep, delta_frame, kmc_time = next(fastforward_gen)
        frame = sweep % trajectory_length

        for i in range(sweep, next_sweep):
            if i % print_frequency == 0:
                print(output_format.format(i, i * timestep_md, *oxygen_trajectory[
                    i % trajectory_length, proton_position], jumps, i / (time.time() - start_time)),
                      flush=True, file=settings.output)
                if DEBUG:
                    print(next(distance_debug)[1][proton_position])
                    print(next(distance_rescaled_debug)[1][proton_position])

        jumps += 1
        sweep = next_sweep
        if DEBUG:
            print("Jumping")
            print("Old proton position:", proton_position)

        probs = kmc_gen.prob
        cumsum = np.cumsum(probs)
        neighbor_indices = indices[frame, proton_position]

        if DEBUG:
            print("Choose between", neighbor_indices)
            print("With probabilities", probs)

        random_draw = np.random.uniform(0, cumsum[-1])
        ix = np.searchsorted(cumsum, random_draw)
        proton_position = neighbor_indices[ix]
        kmc_gen.oxy_idx = proton_position
        # After a jump, the relaxation time is increased
        kmc_gen.relaxation_time = relaxation_time

        if DEBUG:
            print("New proton position:", proton_position)


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
