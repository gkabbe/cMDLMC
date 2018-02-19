import sys
import ipdb
import numpy as np
from collections import Counter
import argparse

from mdlmc.IO import trajectory_parser
from mdlmc.cython_exts.LMC.LMCHelper import AtomBoxCubic, AtomBoxMonoclinic

def main(*args):
    parser = argparse.ArgumentParser(description="topology",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pbc", nargs="+", type=float, help="PBC")
    parser.add_argument("trajectory", help="Trajectory")
    parser.add_argument("--selection", nargs="+", help="Atom names")
    args = parser.parse_args()

    fname = args.trajectory
    selection = args.selection
    c = Counter(selection)

    pbc = np.array(args.pbc)
    if len(pbc) == 3:
        atom_box = AtomBoxCubic(pbc)
    else:
        atom_box = AtomBoxMonoclinic(pbc)

    for k, v in c.items():
        print("Looking for", v, "atoms of type", k)

    frames = dict()
    atom_types = list(c.keys())
    atoms = trajectory_parser.load_atoms(fname, *atom_types, clip=1)

    for a, f in zip(atom_types, atoms):
        frames[a] = f

    ipdb.set_trace()
