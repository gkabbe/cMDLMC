import tables
import h5py
import os

from numba import jit

from mdlmc.IO.xyz_parser import load_trajectory_from_hdf5, save_trajectory_to_hdf5


def get_hdf5_filename(trajectory_fname):
    root, ext = os.path.splitext(trajectory_fname)
    if ext == ".hdf5":
        return trajectory_fname
    else:
        return root + "_nobackup.hdf5"


class KMC:
    def __init__(self, trajectory_fname, *, verbose=False):

        hdf5_fname = get_hdf5_filename(trajectory_fname)
        if os.path.exists(hdf5_fname):
            hdf5_file = h5py.File(hdf5_fname, "w")
        else:
            save_trajectory_to_hdf5(trajectory_fname, hdf5_fname, remove_com_movement=True,
                                    verbose=verbose)

        trajectory = hdf5_file["trajectory"]
        atom_names = hdf5_file["atom_names"][:].astype("U")
        oxygen_shape = (trajectory.shape[0], *trajectory[0][atom_names == "O"].shape)
        print(oxygen_shape)

        if "oxygen_trajectory" not in hdf5_file.keys():
            shape = hdf5_file["trajectory"].shape
            oxygens = hdf5_file.create_dataset("oxygen_trajectory", shape=,
                                        dtype=float, maxshape=(None, *frame_shape),
                                        compression=32001)


def main(*args):
    kmc = KMC()
