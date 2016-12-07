import os

import tables
import h5py
import numpy as np
from numba import jit

from mdlmc.IO.xyz_parser import save_trajectory_to_hdf5
from mdlmc.misc.tools import chunk


def get_hdf5_filename(trajectory_fname):
    root, ext = os.path.splitext(trajectory_fname)
    if ext == ".hdf5":
        return trajectory_fname
    else:
        return root + "_nobackup.hdf5"


@jit(nopython=True)
def determine_probability_sums(a, b, oxygen_trajectory, pbc):
    probs = np.zeros(oxygen_trajectory.shape[:2])

    for f in range(oxygen_trajectory.shape[0]):
        for i in range(oxygen_trajectory.shape[1]):
            for j in range(oxygen_trajectory.shape[1]):
                dist = oxygen_trajectory[f, j] - oxygen_trajectory[f, i]
                pbc_dist(dist, pbc)
                probs[i] += exponential_jumprate(a, b, np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2))


@jit
def pbc_dist(dist, pbc):
    for i in range(3):
        while (dist[i] > pbc[i] / 2):
            dist[i] -= pbc[i]
        while (dist[i] < -pbc[i] / 2):
            dist[i] += pbc[i]


@jit
def exponential_jumprate(a, b, distance):
    return a * np.exp(b * distance)


def main(*args):
    kmc = KMC("400Kbeginning.xyz", pbc=np.array([29.122, 25.354, 12.363]), verbose=True)

    step = 10000

    hdf5_fname = get_hdf5_filename(trajectory_fname)
    if verbose:
        print("Looking for HDF5 file", hdf5_fname)
    if not os.path.exists(hdf5_fname):
        if verbose:
            print("Could not find HDF5 file. Will create it now.")
        save_trajectory_to_hdf5(trajectory_fname, hdf5_fname, remove_com_movement=True,
                                verbose=verbose)

    hdf5_file = h5py.File(hdf5_fname, "a")
    trajectory = hdf5_file["trajectory"]

    atom_names = hdf5_file["atom_names"][:].astype("U")
    oxygen_mask = atom_names == "O"
    oxygen_shape = (trajectory.shape[0], *trajectory[0][oxygen_mask].shape)
    print(oxygen_shape)

    if "oxygen_trajectory" not in hdf5_file.keys():
        print("Found no oxygen trajectory")

        oxygens = hdf5_file.create_dataset("oxygen_trajectory", shape=oxygen_shape,
                                           dtype=np.float32, compression=32001)

        for start, stop, _ in chunk(range(oxygens.shape[0]), step):
            oxygens[start:stop] = trajectory[start:stop, oxygen_mask]

    oxygens = hdf5_file["oxygen_trajectory"]

    a, b = 1.132396e+15, -16.001675
