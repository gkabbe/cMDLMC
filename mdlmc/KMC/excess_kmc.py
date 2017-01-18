import argparse
import os
import time

import tables
import h5py
import numpy as np

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5, create_dataset_from_hdf5_trajectory
from mdlmc.IO.config_parser import load_configfile, print_settings, print_config_template, print_confighelp
from mdlmc.misc.tools import chunk, argparse_compatible
from mdlmc.cython_exts.LMC.LMCHelper import KMCRoutine, FermiFunction
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxWater, AtomBoxWaterLinearConversion, \
    AtomBoxWaterRampConversion
from mdlmc.LMC.MDMC import initialize_oxygen_lattice


def fermi(x, a, b, c):
    return a / (1 + np.exp(-(x - b) / c))


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


def fastforward_to_next_jump(probsums, proton_position, dt, frame, time, traj_len):
    """Implements Kinetic Monte Carlo with time-dependent rates.

    Parameters
    ----------
    probsums : array_like
        Array containing the probability of a proton jump for every possible proton position
        at every frame in the trajectory. Shape: (Trajectory length, No. of oxygen atoms)
        Unit: femtosecond^{-1}
    proton_position : int
        Index of oxygen at which the excess proton is residing.
    dt : float
        Trajectory time step
    frame : int
        Frame index of trajectory
    time : float
        Time passed
    traj_len: int
        Trajectory length

    Returns
    -------
    delta_frame : int
        Difference between frame and the index at which the next event occurs
    delta_t : float
        Difference between current time and the time of the next event
    """

    # Arbitrary guess
    relaxation_frames = 200

    time_selector = -np.log(1 - np.random.random())

    # Handle case where time selector is so small that the next frame is not reached
    t_trial = time_selector / probsums[frame, proton_position]
    if (time + t_trial) // dt == time // dt:
        return 0, t_trial

    delta_t, delta_frame = dt - time % dt, 1
    current_prob = probsums[frame, proton_position] * delta_t
    next_frame = frame + 1
    if next_frame == traj_len:
        next_frame = 0
    next_prob = current_prob + probsums[next_frame, proton_position] * dt

    while next_prob < time_selector:
        delta_frame += 1
        frame = next_frame
        next_frame = frame + 1
        if next_frame == traj_len:
            next_frame = 0
        current_prob = next_prob
        next_prob = current_prob + probsums[next_frame, proton_position] * dt

    rest = time_selector - current_prob
    delta_t += (delta_frame - 1) * dt + rest / probsums[frame, proton_position]
    return delta_frame, delta_t


def trajectory_generator(hdf5_dataset, chunk_size=10000):
    while True:
        for start, stop, chk in chunk(hdf5_dataset, chunk_size=chunk_size):
            for frame in chk:
                yield frame


class KMCGen:
    def __init__(self, oxy_idx, distances, distances_rescaled, jumprate_fct, jumprate_params):
        self.oxy_idx = oxy_idx
        self.delay = 0
        self.distances = distances
        self.distances_rescaled = distances_rescaled
        self.jumprate_fct = jumprate_fct
        self.jumprate_params = jumprate_params

    def distance_generator(self):
        distance_gen = trajectory_generator(self.distances)
        distance_rescaled_gen = trajectory_generator(self.distances_rescaled)

        while True:
            if self.delay:
                for i in range(self.delay):
                    dist = next(distance_gen)[self.oxy_idx]
                    dist_rescaled = next(distance_rescaled_gen)[self.oxy_idx]
                    yield dist + i / (self.delay - 1) * (dist_rescaled - dist)
                self.delay = 0
            else:
                yield next(distance_rescaled_gen)[self.oxy_idx]

    def probsum_generator(self):
        distance_gen = self.distance_generator()
        while True:
            for dists in distance_gen:
                yield self.jumprate_fct(dists, *self.jumprate_params)


def kmc_main(settings):
    print_settings(settings)

    trajectory_fname = settings.filename
    verbose = settings.verbose
    chunk_size = settings.chunk_size

    hdf5_fname = get_hdf5_filename(trajectory_fname)
    if verbose:
        print("# Looking for HDF5 file", hdf5_fname)
    if not os.path.exists(hdf5_fname):
        if verbose:
            print("# Could not find HDF5 file. Will create it now.")
        save_trajectory_to_hdf5(trajectory_fname, hdf5_fname, remove_com_movement=True,
                                verbose=verbose)

    hdf5_file = h5py.File(hdf5_fname, "a")
    trajectory = hdf5_file["trajectory"]

    atom_names = hdf5_file["atom_names"][:].astype("U")

    oxygen_indices, = np.where(atom_names == "O")
    if verbose:
        print("# Loading oxygen trajectory", file=settings.output)
    oxygen_trajectory = create_dataset_from_hdf5_trajectory(hdf5_file, trajectory, "oxygen_trajectory",
                                                            oxygen_indices, chunk_size)

    proton_indices = np.where(atom_names == "H")
    if verbose:
        print("# Loading proton trajectory", file=settings.output)
    proton_trajectory = create_dataset_from_hdf5_trajectory(hdf5_file, trajectory, "proton_trajectory",
                                                            proton_indices, chunk_size)

    trajectory_length = oxygen_trajectory.shape[0]
    current_time, timestep_md = 0, settings.md_timestep_fs
    frame, sweep, jumps = 0, 0, 0
    total_sweeps = settings.sweeps
    xyz_output = settings.xyz_output
    print_frequency = settings.print_frequency

    oxygen_number = len(oxygen_indices)
    # Initialize with one excess proton
    oxygen_lattice = initialize_oxygen_lattice(oxygen_number, 1)
    proton_position, = np.where(oxygen_lattice)[0]

    atombox_cubic = AtomBoxCubic(settings.pbc)

    if settings.rescale_parameters:
        if settings.rescale_function == "linear":
            atombox_rescale = AtomBoxWaterLinearConversion(settings.pbc, settings.rescale_parameters)
        elif settings.rescale_function == "ramp_function":
            atombox_rescale = AtomBoxWaterRampConversion(settings.pbc, settings.rescale_parameters)
        else:
            raise ValueError("Unknown rescale function name", settings.rescale_function)

    a, b, c = settings.jumprate_params_fs["a"], settings.jumprate_params_fs["b"], \
              settings.jumprate_params_fs["c"]

    jumprate_function = FermiFunction(a, b, c)
    kmc = KMCRoutine(atombox_cubic, oxygen_lattice, jumprate_function)
    if settings.rescale_parameters:
        kmc_rescale = KMCRoutine(atombox_rescale, oxygen_lattice, jumprate_function)

    if settings.rescale_parameters:
        distname = "distances_rescaled"
        rescaled = "rescaled "
    else:
        distname = "distances"
        rescaled = "unrescaled"

    if verbose:
        print("# Creating array of distances", file=settings.output)
    distances, indices = create_dataset_from_hdf5_trajectory(hdf5_file, oxygen_trajectory,
                                                             ("distances", "indices"),
                                                             kmc.determine_distances,
                                                             chunk_size, dtype=(float, int),
                                                             overwrite=settings.overwrite_jumprates)

    if settings.rescale_parameters:
        if verbose:
            print("# Creating array of rescaled distances")
        distances_rescaled, indices = create_dataset_from_hdf5_trajectory(
            hdf5_file, oxygen_trajectory, ("distances_rescaled", "indices"),
            kmc_rescale.determine_distances, chunk_size, dtype=(float, int),
            overwrite=settings.overwrite_jumprates)

    if verbose:
        print("# Creating probability sums {}".format(rescaled), file=settings.output)

    probsums = create_dataset_from_hdf5_trajectory(
        hdf5_file, distances, "probability_sums".format(rescaled),
        lambda x: np.sum(fermi(x, a, b, c), axis=-1), chunk_size,
        overwrite=settings.overwrite_jumprates)

    if settings.rescale_parameters:
        probsums_rescaled = create_dataset_from_hdf5_trajectory(
            hdf5_file, distances_rescaled, "probability_sums_rescaled".format(rescaled),
            lambda x: np.sum(fermi(x, a, b, c), axis=-1), chunk_size,
            overwrite=settings.overwrite_jumprates)

    output_format = "{:18d} {:18.2f} {:15.8f} {:15.8f} {:15.8f} {:10d}"

    while sweep < total_sweeps:

        delta_frame, delta_time = fastforward_to_next_jump(probsums, proton_position, timestep_md,
                                                           frame, current_time, trajectory_length)

        for i in range(sweep, sweep + delta_frame):
            if i % print_frequency == 0:
                if xyz_output:
                    kmc_state_to_xyz(oxygen_trajectory[i % trajectory_length],
                                     proton_trajectory[i % trajectory_length], oxygen_lattice)
                else:
                    print(output_format.format(i, i * timestep_md,
                                               *oxygen_trajectory[i % trajectory_length, proton_position],
                                               jumps), flush=True, file=settings.output)

        frame = (frame + delta_frame) % trajectory_length
        sweep += delta_frame
        current_time += delta_time
        jumps += 1

        probs = np.cumsum(probabilities[frame, proton_position])
        neighbor_indices = indices[frame, proton_position]
        random_draw = np.random.uniform(0, probs[-1])
        ix = np.searchsorted(probs, random_draw)
        proton_position = neighbor_indices[ix]

        if sweep % print_frequency == 0:
            if xyz_output:
                kmc_state_to_xyz(oxygen_trajectory[frame], proton_trajectory[frame],
                                 oxygen_lattice)
            else:
                print(output_format.format(sweep, current_time, *oxygen_trajectory[frame, proton_position],
                                           jumps), flush=True, file=settings.output)


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
