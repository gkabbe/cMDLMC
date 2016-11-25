#!/usr/bin/env python3

import os
import time

import numpy as np

from mdlmc.atoms import numpyatom as npa
from mdlmc.atoms.numpyatom import dtype_xyz, dtype_xyz_bytes


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


def save_trajectory_to_hdf5(xyz_fname, hdf5_fname=None, chunk=1000, *, remove_com_movement=False,
                            verbose=False):
    import tables
    import h5py
    with open(xyz_fname, "rb") as f:
        frame_len = int(f.readline()) + 2
        f.seek(0)
        first_frame, = parse_xyz(f, frame_len, no_of_frames=1)

    if not hdf5_fname:
        hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with h5py.File(hdf5_fname, "w") as hdf5_file:
        # Use blosc compression (needs tables import and code 32001)
        traj = hdf5_file.create_dataset("trajectory", shape=first_frame.shape,
                                        dtype=dtype_xyz_bytes,
                                        maxshape=(None, *first_frame.shape[1:]), compression=32001)

        with open(xyz_fname, "rb") as f:
            counter = 0
            start_time = time.time()
            while True:
                frames = parse_xyz(f, frame_len, no_of_frames=chunk)
                if frames.shape[0] == 0:
                    break
                if remove_com_movement:
                    npa.remove_center_of_mass_movement(frames)
                # import ipdb
                # ipdb.set_trace()
                traj[counter:counter + chunk] = frames
                counter += chunk
                print("# Parsed frames: {:06d}. {:.2f} fps".format(
                    counter, counter / (time.time() - start_time)), end="\r")


def load_trajectory_from_hdf5(hdf5_fname, *atom_names, clip=None, verbose=False):
    import tables
    f = tables.open_file(hdf5_fname, "r")
    slice_ = slice(0, clip)
    trajectories = [f.get_node("/trajectories", atom_name)[slice_] for atom_name in atom_names]
    return trajectories


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
