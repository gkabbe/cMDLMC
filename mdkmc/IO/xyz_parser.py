#!/usr/bin/env python3

import os
import sys
import time
import re

import ipdb
import numpy as np
import tables

from mdkmc.atoms import numpyatom as npa
from mdkmc.atoms.numpyatom import xyzatom as dtype_xyz


def parse(f, frame_len, *atom_names, no_of_atoms=None, no_of_frames=None):
    def filter_lines(f, frame_len):
        for i, line in enumerate(f):
            if i % frame_len not in (0, 1):
                yield line
                if (i + 1) // frame_len == no_of_frames:
                    break

    def filter_atoms(f, atom_names):
        for line in f:
            if line.split()[0].decode("utf-8") in atom_names:
                yield line

    if len(atom_names) > 0:
        filter_ = filter_atoms(filter_lines(f, frame_len), atom_names)
        output_shape = (-1, no_of_atoms)
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


