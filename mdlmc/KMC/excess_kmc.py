import argparse
import os
import configparser

import tables
import h5py
import numpy as np

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5
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


@argparse_compatible
def kmc(config_file):
    settings = load_configfile(config_file, config_type="KMCWater")
    print_settings(settings)

    trajectory_fname = settings.filename
    verbose = settings.verbose
    atombox = AtomBoxCubic(settings.pbc)
    a, b = 1.132396e+15, -16.001675

    jumprate_function = ExponentialFunction(a, b)
    kmc = KMCRoutine(trajectory_fname, atombox, jumprate_function)

    chunk_size = 10000

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
    oxygen_mask = atom_names == "O"
    oxygen_shape = (trajectory.shape[0], *trajectory[0][oxygen_mask].shape)
    print(oxygen_shape)

    if "oxygen_trajectory" not in hdf5_file.keys():
        print("# Found no oxygen trajectory in HDF5 file")
        oxygens = hdf5_file.create_dataset("oxygen_trajectory", shape=oxygen_shape,
                                           dtype=float, compression=32001)
        oxygens[:] = np.nan

    else:
        oxygens = hdf5_file["oxygen_trajectory"]

    if np.isnan(hdf5_file["oxygen_trajectory"]).any():
        print("# It looks like the oxygen trajectory was not written completely to hdf5")
        print("# Will do it now")
        for start, stop, _ in chunk(range(oxygens.shape[0]), chunk_size):
            oxygens[start:stop] = trajectory[start:stop, oxygen_mask]

    if "probability_sums" not in hdf5_file.keys():
        print("# Found no probability sums in HDF5 file")
        probsums = hdf5_file.create_dataset("probability_sums", shape=(oxygen_shape[:2]), dtype=float,
                                            compression=32001)
    else:
        probsums = hdf5_file["probability_sums"]

    if np.isnan(hdf5_file["probability_sums"]).any():
        print("# It looks like the probability sums were not written completely to hdf5")
        print("# Will do it now")
        for start, stop, oxy_chunk in chunk(oxygens, 100):
            probsums[start:stop] = kmc.determine_probability_sums(oxy_chunk)


def main(*args):
    parser = argparse.ArgumentParser(description="KMC",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", help="Config file")
    parser.set_defaults(func=kmc)
    args = parser.parse_args()
    args.func(args)
