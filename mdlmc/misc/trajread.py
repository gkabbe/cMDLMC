import sys
from mdlmc.IO import xyz_parser
from mdlmc.atoms import numpyatom as npa


def main(*args):
    filename = sys.argv[1]
    trajectory = xyz_parser.load_trajectory_from_npz(filename)

    for frame in trajectory:
        npa.numpy_print(frame)
