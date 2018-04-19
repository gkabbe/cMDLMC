import logging
import os
import pathlib

import tables
import h5py
import daiquiri
import fire
import numpy as np
from typing import Union, Iterable

from ..atoms.numpy_atom import dtype_xyz
from ..atoms import numpy_atom as npa
from ..IO.trajectory_parser import XYZTrajectory


logger = logging.getLogger(__name__)


def save_xyz_to_hdf5(xyz_fname, hdf5_fname=None, *, remove_com_movement=False,
                     dataset_name="trajectory", selection=None):
    """
    Note: HDF5 with Blosc compression currently only works if h5py and pytables are installed via
    conda!"""
    xyz = XYZTrajectory(xyz_fname, selection=selection)
    logger.info("Determine length of xyz trajectory.")
    trajectory_length = len(xyz)
    first_frame = next(iter(xyz))
    frame_shape = first_frame.atom_positions.shape
    atom_names = first_frame.atom_names.astype("S")
    logger.info("Names: %s", atom_names)

    if not hdf5_fname:
        hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with h5py.File(hdf5_fname, "w") as hdf5_file:
        # Use blosc compression (needs tables import and code 32001)
        traj_atomnames = hdf5_file.create_dataset("atom_names", atom_names.shape, dtype="2S")
        traj_atomnames[:] = atom_names
        traj = hdf5_file.create_dataset(dataset_name, shape=(trajectory_length, *frame_shape),
                                        dtype=np.float32, compression=32001)

        for i, xyz_frame in enumerate(xyz):
            if remove_com_movement:
                npa.remove_center_of_mass_movement(xyz_frame)
            traj[i] = xyz_frame.atom_positions


def main():
    daiquiri.setup(level=logging.INFO)
    fire.Fire()
