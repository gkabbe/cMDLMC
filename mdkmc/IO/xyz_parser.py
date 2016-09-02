#!/usr/bin/env python3

import os
import sys
import time
import re
from functools import reduce

import ipdb
import numpy as np
import tables

from mdkmc.atoms import numpyatom as npa
from mdkmc.atoms.numpyatom import xyzatom as dtype_xyz


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

    data = np.genfromtxt(filter_, dtype=dtype_xyz)
    return data.reshape(output_shape)


def save_trajectory_to_hdf5(xyz_fname, *atom_names, only_acidic_protons=False, pbc=None,
                            hdf5_fname=None, chunk=1000,
                            remove_com_movement=False, verbose=False):
    with open(xyz_fname, "rb") as f:
        frame_len = int(f.readline()) + 2
        f.seek(0)
        first_frame, = parse_xyz(f, frame_len, no_of_frames=1)
        if "H" in atom_names and only_acidic_protons and pbc is not None:
            acidic_proton_selection = np.array(npa.get_acidic_proton_indices(first_frame, pbc))
            selections = dict(H=acidic_proton_selection)
        else:
            selections = dict()
        for atom_name in atom_names:
            atom_selection = np.where(first_frame["name"] == atom_name)[0]
            selections[atom_name] = atom_selection

        all_indices = reduce(set.union, [set(s) for s in selections.values()])

    a = tables.Atom.from_dtype(np.dtype("float"))
    filters = tables.Filters(complevel=5, complib="blosc")
    if not hdf5_fname:
        hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with tables.open_file(hdf5_fname, "a", filters=filters) as hdf5:
        hdf5.create_group("/", "trajectories")
        for atom_name, indices in selections.items():
            hdf5.create_earray("/trajectories", atom_name, atom=a, shape=(0, len(indices), 3))

        with open(xyz_fname, "rb") as f:
            counter = 0
            start_time = time.time()
            while True:
                frames = parse_xyz(f, frame_len, no_of_frames=chunk)
                if frames.shape[0] == 0:
                    break
                if remove_com_movement:
                    for frame in frames:
                        npa.remove_com_movement_frame(frame)
                for atom_name, indices in selections.items():
                    hdf5.get_node("/trajectories", atom_name).append(frames[:, frames[0]["name"] == atom_name]["pos"])
                counter += chunk
                print("# Parsed frames: {:06d}. {:.2f} fps".format(
                    counter, counter / (time.time() - start_time)), end="\r")


def load_trajectory_from_hdf5(hdf5_fname, *atom_names, clip=None, verbose=False):
    f = tables.open_file(hdf5_fname, "r")
    slice_ = slice(0, clip)
    trajectories = [f.get_node("/trajectories", atom_name)[slice_] for atom_name in atom_names]
    return trajectories


def save_trajectory_to_npz(xyz_fname, npz_fname=None, remove_com_movement=False,
                           verbose=False):
    with open(xyz_fname, "rb") as f:
        frame_length = int(f.readline()) + 2
        f.seek(0)
        if verbose:
            print("# Determining trajectory length...")
        f.seek(0)

        data = parse_xyz(f, frame_length)

        if remove_com_movement:
            npa.remove_center_of_mass_movement_fast(data)

        if not npz_fname:
            npz_fname = os.path.splitext(xyz_fname)[0] + ".npy"
        np.savez(npz_fname, trajectory=data)


def load_trajectory_from_npz(npz_fname, *atom_names, clip=None, verbose=False):
    trajectory = np.load(npz_fname)["trajectory"]
    if clip:
        if verbose:
            print("# Clipping trajectory to the first {} frames".format(clip))
        trajectory = trajectory[:clip]
    single_atom_trajs = []
    for atom in atom_names:
        atom_traj = trajectory[:, trajectory[0]["name"] == atom]
        atom_traj = np.array(atom_traj["pos"], order="C")
        single_atom_trajs.append(atom_traj)
    return single_atom_trajs
