import argparse
import os
import configparser

import tables
import h5py
import numpy as np

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5, create_dataset_from_hdf5_trajectory
from mdlmc.IO.config_parser import load_configfile, print_settings
from mdlmc.misc.tools import chunk, argparse_compatible
from mdlmc.cython_exts.LMC.LMCHelper import KMCRoutine, ExponentialFunction
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


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


@argparse_compatible
def kmc(config_file):
    settings = load_configfile(config_file, config_type="KMCWater")
    print_settings(settings)

    trajectory_fname = settings.filename
    verbose = settings.verbose
    atombox = AtomBoxCubic(settings.pbc)
    a, b = settings.jumprate_params_fs["a"], settings.jumprate_params_fs["b"]
    jumprate_function = ExponentialFunction(a, b)
    kmc = KMCRoutine(trajectory_fname, atombox, jumprate_function)
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
    oxygen_trajectory = create_dataset_from_hdf5_trajectory(hdf5_file, trajectory, "oxygen_trajectory",
                                                            oxygen_indices, chunk_size)

    proton_indices = np.where(atom_names == "H")
    proton_trajectory = create_dataset_from_hdf5_trajectory(hdf5_file, trajectory, "proton_trajectory",
                                                            proton_indices, chunk_size)

    probsums = create_dataset_from_hdf5_trajectory(hdf5_file, oxygen_trajectory, "probability_sums",
                                                   kmc.determine_probability_sums, chunk_size)

    trajectory_length = oxygen_trajectory.shape[0]
    t, dt = 0, settings.md_timestep_fs
    frame, sweep, jumps = 0, 0, 0
    total_sweeps = settings.sweeps
    xyz_output = settings.xyz_output

    if xyz_output:
        proton_trajectory, = load_atoms(settings.filename, "H")

    output_format = "{:18d} {:18.2f} {:15.8f} {:15.8f} {:15.8f} {:10d}"

    while sweep < total_sweeps:
        proton_position = np.where(oxygen_lattice)[0][0]
        time_selector = -np.log(np.random.random())
        prob_sum = 0
        start, destination, probabilities = helper.return_transitions(frame)
        transition_indices, = np.where(np.array(start) == proton_position)
        prob_sum += determine_probsum(probabilities, transition_indices, dt)
        while prob_sum < time_selector:
            if xyz_output:
                kmc_state_to_xyz(oxygen_trajectory[frame], proton_trajectory[frame],
                                 oxygen_lattice)
            else:
                print(output_format.format(sweep, t, *oxygen_trajectory[frame, proton_position],
                                           jumps), flush=True, file=settings.output)
            sweep, t = sweep + 1, t + dt
            frame = sweep % trajectory_length
            start, destination, probabilities = helper.return_transitions(frame)
            transition_indices, = np.where(np.array(start) == proton_position)
            prob_sum += determine_probsum(probabilities, transition_indices, dt)
        jumps += 1
        transition_probs = np.array(probabilities)[transition_indices]
        destination_indices = np.array(destination)[transition_indices]
        event_selector = np.random.random() * transition_probs.sum()
        transition_index = np.searchsorted(np.cumsum(transition_probs), event_selector)
        oxygen_lattice[proton_position] = 0
        proton_position = destination_indices[transition_index]
        oxygen_lattice[proton_position] = 1
        if xyz_output:
            kmc_state_to_xyz(oxygen_trajectory[frame], proton_trajectory[frame],
                             oxygen_lattice)
        else:
            print(output_format.format(sweep, t, *oxygen_trajectory[frame, proton_position],
                                       jumps), flush=True, file=settings.output)


def main(*args):
    parser = argparse.ArgumentParser(description="KMC",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="Config file")
    parser.set_defaults(func=kmc)
    args = parser.parse_args()
    args.func(args)
