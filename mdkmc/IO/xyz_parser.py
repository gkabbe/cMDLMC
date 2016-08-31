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


def save_trajectory_to_hdf5(xyz_fname, *atom_names):
    with open(xyz_fname, "rb") as f:
        atom_number = int(f.readline())
        f.seek(0)
        first_frame = parse(f, atom_number, chunk = 1)
        atom_nr_dict = dict()
        for name in atom_names:
            atom_nr_dict[name] = (first_frame[0]["name"] == name).sum()
    
    a = tables.Atom.from_dtype(np.dtype("float32"))
    filters = tables.Filters(complevel=5, complib="blosc")
    hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with tables.open_file(hdf5_fname, "a") as f:
        f.create_group("/", filters=filters)
        for name in atom_names:
            f.create_earray("/", atom_name, atom=a, shape=(0, atom_nr_dict[name], 3))
