# coding=utf-8

import sys
from mdlmc.IO import trajectory_parser
from mdlmc.atoms import numpy_atom as npa


def main(*args):
    filename = sys.argv[1]
    trajectory = trajectory_parser.load_trajectory_from_npz(filename)

    for frame in trajectory:
        npa.numpy_print(frame)
