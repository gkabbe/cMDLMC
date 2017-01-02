#!/usr/bin/env python3

import argparse
from functools import reduce
import os
import time
import types
import sys

import numpy as np
import tables
import h5py

from mdlmc.atoms import numpyatom as npa
from mdlmc.atoms.numpyatom import dtype_xyz, dtype_xyz_bytes
from mdlmc.misc.tools import argparse_compatible, chunk


def parse_xyz(f, frame_len, selection=None, no_of_frames=None):
    def filter_lines(f, frame_len):
        for i, line in enumerate(f):
            if i % frame_len not in (0, 1):
                yield line
                if (i + 1) // frame_len == no_of_frames:
                    break

    if selection:
        def filter_selection(f, s):
            for i, line in enumerate(f):
                if i % (frame_len - 2) in s:
                    yield line

        filter_ = filter_selection(filter_lines(f, frame_len), selection)
        output_shape = (-1, len(selection))
    else:
        filter_ = filter_lines(f, frame_len)
        output_shape = (-1, frame_len - 2)

    data = np.genfromtxt(filter_, dtype=dtype_xyz_bytes)
    return data.reshape(output_shape)


@argparse_compatible
def save_trajectory_to_hdf5(xyz_fname, hdf5_fname=None, chunk=1000, *, remove_com_movement=False,
                            verbose=False, file=sys.stdout):

    with open(xyz_fname, "rb") as f:
        frame_len = int(f.readline()) + 2
        f.seek(0)
        first_frame, = parse_xyz(f, frame_len, no_of_frames=1)
        atom_names = first_frame["name"]
        frame_shape = first_frame["pos"].shape

    if not hdf5_fname:
        hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with h5py.File(hdf5_fname, "w") as hdf5_file:
        # Use blosc compression (needs tables import and code 32001)
        traj_atomnames = hdf5_file.create_dataset("atom_names", atom_names.shape, dtype="2S")
        traj_atomnames[:] = atom_names
        traj = hdf5_file.create_dataset("trajectory", shape=(10 * chunk, *frame_shape),
                                        dtype=float, maxshape=(None, *frame_shape),
                                        compression=32001)

        with open(xyz_fname, "rb") as f:
            counter = 0
            start_time = time.time()
            while True:
                frames = parse_xyz(f, frame_len, no_of_frames=chunk).astype(dtype_xyz)

                if frames.shape[0] == 0:
                    break

                if remove_com_movement:
                    npa.remove_center_of_mass_movement(frames)

                # Resize trajectory if necessary
                if counter + frames.shape[0] > traj.shape[0]:
                    if verbose:
                        print("# Need to resize trajectory hdf5 dataset", file=file)
                    traj.resize(2 * traj.shape[0], axis=0)

                traj[counter:counter + frames.shape[0]] = frames["pos"]
                counter += frames.shape[0]
                print("# Parsed frames: {: 6d}. {:.2f} fps".format(
                    counter, counter / (time.time() - start_time)), end="\r", flush=True, file=file)
        traj.resize(counter, axis=0)


def load_trajectory_from_hdf5(hdf5_fname, *atom_names, clip=None, verbose=False, file=sys.stdout):
    with h5py.File(hdf5_fname, "r") as f:
        traj_atom_names = f["atom_names"].value.astype("U")
        if atom_names:
            if verbose:
                print("# Will select atoms", *atom_names, file=file)
            selection = reduce(np.logical_or, [traj_atom_names == name for name in atom_names])
        else:
            selection = slice(None)
        slice_ = slice(0, clip)
        trajectories = f["trajectory"][slice_, selection]
    return traj_atom_names[selection], trajectories


def create_dataset_from_hdf5_trajectory(hdf5_file, trajectory_dataset, dataset_name, selection,
                                        chunk_size, verbose=False, file=sys.stdout):
    """Given a hdf5 trajectory, this function creates a new data set inside the hdf5 file with the
    specified dataset_name according to the supplied selection.
    If it already exists, it will be returned directly.

    Parameters
    ----------
    hdf5_file: HDF5 file
               The new dataset will be saved in this file

    trajectory_dataset: array_like
                        The array or HDF5 dataset containing the atom trajectory

    dataset_name: str
                  Name of the new dataset

    selection: function or array_like
               Will be used to select items from trajectory_dataset.
               Can be an array of indices or a function that creates a new array out of
               trajectory_dataset

    Returns
    -------
    new_dataset: HDF5 dataset"""

    first_frame = trajectory_dataset[:1]
    if type(selection) in (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                           types.BuiltinMethodType):
        selection_fct = selection
    else:
        def selection_fct(arr):
            return np.take(arr, selection, axis=1)

    one_frame_shape = selection_fct(first_frame).shape

    if dataset_name not in hdf5_file.keys():
        print("# Did not find data set", dataset_name, "in HDF5 file")
        new_dataset = hdf5_file.create_dataset(dataset_name,
                                               shape=(trajectory_dataset.shape[0], *one_frame_shape[1:]),
                                               dtype=float, compression=32001)
        # Set last frame to nan, to identify the data set as unfinished
        new_dataset[-1] = np.nan

    else:
        new_dataset = hdf5_file[dataset_name]

    if np.isnan(hdf5_file[dataset_name][-1]).any():
        print("# It looks like data set", dataset_name, "has not been written to hdf5 yet.",
              file=file)
        print("# Will do it now", file=file)
        start_time = time.time()
        for start, stop, traj_chunk in chunk(trajectory_dataset, chunk_size):
            new_dataset[start:stop] = selection_fct(traj_chunk)
            print("# Parsed frames: {: 6d}. {:.2f} fps".format(
                  stop, stop / (time.time() - start_time)), end="\r", flush=True, file=file)

    return new_dataset


def save_trajectory_to_npz(xyz_fname, npz_fname=None, remove_com_movement=False,
                           verbose=False):
    with open(xyz_fname, "rb") as f:
        frame_length = int(f.readline()) + 2
        f.seek(0)

        chunk_size = 1000
        counter = 0
        trajectory = []
        start_time = time.time()
        data = parse_xyz(f, frame_length, no_of_frames=chunk_size)
        while data.shape[0] > 0:
            counter += chunk_size
            fps = counter / (time.time() - start_time)
            print("# {:6d} ({:.2f} fps)".format(counter, fps), end="\r", flush=True)
            trajectory.append(data)
            data = parse_xyz(f, frame_length, no_of_frames=chunk_size)
        print("")
        trajectory = np.vstack(trajectory)

        if remove_com_movement:
            if verbose:
                print("# Removing center of mass movement...")
            npa.remove_center_of_mass_movement(trajectory)

        if not npz_fname:
            npz_fname = os.path.splitext(xyz_fname)[0] + ".npy"
        np.savez(npz_fname, trajectory=trajectory)


def load_trajectory_from_npz(npz_fname, *atom_names, clip=None, verbose=False):
    file_content = np.load(npz_fname)
    if type(file_content) == np.ndarray:
        trajectory = file_content
    else:
        if "trajectory" in file_content.keys():
            trajectory = file_content["trajectory"]
        else:
            trajectory = file_content.items()[0]

    if clip:
        if verbose:
            print("# Clipping trajectory to the first {} frames".format(clip))
        trajectory = trajectory[:clip]
    if len(atom_names) > 0:
        single_atom_trajs = []
        for atom in atom_names:
            atom_traj = trajectory[:, trajectory[0]["name"] == atom]
            atom_traj = np.array(atom_traj["pos"], order="C")
            single_atom_trajs.append(atom_traj)
        return single_atom_trajs
    else:
        return trajectory


def load_atoms(filename, *atom_names, auxiliary_file=None, verbose=False, clip=None, hdf5=False):

    if auxiliary_file:
        # User explicitly specified auxiliary file
        if verbose:
            print("# Both xyz file and auxiliary npz/npy file specified.")
            print("# Will try to load from auxiliary file", auxiliary_file)
    else:
        # Look for auxiliary file with same name as xyz file, but different extension
        if hdf5:
            auxiliary_file = os.path.splitext(filename)[0] + ".hdf5"
        else:
            auxiliary_file = os.path.splitext(filename)[0] + ".npz"

    if verbose:
        print("# Looking for auxiliary file", auxiliary_file, "...")

    if os.path.exists(auxiliary_file):
        if verbose:
            print("# Found it!")

    else:

        if verbose:
            print("# No auxiliary file found.")
            print("# Will create it now...")
        if hdf5:
            save_trajectory_to_hdf5(filename, hdf5_fname=auxiliary_file,
                                    remove_com_movement=True, verbose=verbose)
        else:
            save_trajectory_to_npz(filename, npz_fname=auxiliary_file,
                                   remove_com_movement=True, verbose=verbose)

    if hdf5:
        return load_trajectory_from_hdf5(auxiliary_file, *atom_names, clip=clip, verbose=verbose)
    else:
        return load_trajectory_from_npz(auxiliary_file, *atom_names, clip=clip, verbose=verbose)


def save_to_hdf5_cmd(*args):
    parser = argparse.ArgumentParser(
        description="Save trajectory to HDF5", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("xyz_fname", help="XYZ trajectory file name")
    parser.add_argument("--hdf5_fname", help="HDF5 file name")
    parser.add_argument("--chunk", "-c", type=int, default=1000, help="Chunk size")
    parser.add_argument("--remove_com", "-r", action="store_true", help="Remove center of mass movement")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    parser.set_defaults(func=save_trajectory_to_hdf5)
    args = parser.parse_args()
    args.func(args)


def print_hdf5(*args):
    parser = argparse.ArgumentParser(description="Print HDF5 trajectory to command line in xyz "
                                                 "format",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hdf5_fname", help="HDF5 file name")
    parser.add_argument("--atom_names", "-a", nargs="*", default=[],
                        help="Which atoms should be printed?")
    parser.add_argument("--clip", "-c", type=int, help="Clip trajectory after number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    args = parser.parse_args()
    atom_names, trajectory = load_trajectory_from_hdf5(args.hdf5_fname, *args.atom_names,
                                                       clip=args.clip, verbose=args.verbose)

    for frame in trajectory:
        print(frame.shape[0])
        print()
        for name, pos in zip(atom_names, frame):
            print(name, " ".join(map(str, pos)))
