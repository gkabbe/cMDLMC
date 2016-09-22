#!/usr/bin/env python3 -u

import sys
import numpy as np
import time
import argparse
import os
import re

from mdlmc.IO import xyz_parser
from mdlmc.cython_exts.atoms import numpyatom as cnpa
from mdlmc.cython_exts.LMC.LMCHelper import AtomBoxMonoclin, AtomBoxCubic


def determine_phosphorus_oxygen_pairs(oxygen_frame, phosphorus_frame, pbc):
    P_neighbors = np.zeros(oxygen_frame.shape[0], int)
    for i in range(oxygen_frame.shape[0]):
        P_index = cnpa.next_neighbor(oxygen_frame[i], phosphorus_frame, pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors


def determine_covalently_bonded_oxygens(oxygen_trajectory, proton_trajectory, pbc, *,
                                        nonorthogonal_box=False, verbose=False):
    """Saves for each proton at every time step the index of its closest oxygen neighbor"""

    print("# Determining covalently bonded oxygens over time..")
    start_time = time.time()
    covalently_bonded_oxygens = np.zeros((proton_trajectory.shape[0], proton_trajectory.shape[1]),
                                         dtype=int)

    if nonorthogonal_box:
        if verbose:
            print("# Nonorthogonal periodic boundary conditions")
        atom_box = AtomBoxMonoclin(pbc)
    else:
        if verbose:
            print("# Cubic periodic boundary conditions")
        atom_box = AtomBoxCubic(pbc)

    for i in range(covalently_bonded_oxygens.shape[0]):
        for j in range(covalently_bonded_oxygens.shape[1]):
            covalently_bonded_oxygens[i, j], _ = \
                atom_box.next_neighbor(proton_trajectory[i, j], oxygen_trajectory[i])
        if verbose and i % 100 == 0:
            print("# Frame {: 6d} ({:5.0f} fps)".format(i, float(i) / (time.time() - start_time)),
                  end='\r')
    print("")

    return covalently_bonded_oxygens


def jump_histo(dmin, dmax, bins, pbc, oxygen_trajectory, proton_trajectory,
               covalently_bonded_oxygens, verbose=False, nonorthogonal_box=False):
    """Counts proton jumps at different distances"""
    start_time = time.time()

    if nonorthogonal_box:
        atom_box = AtomBoxMonoclin(pbc)
    else:
        atom_box = AtomBoxCubic(pbc)

    jump_counter = np.zeros(bins, int)

    for frame in range(oxygen_trajectory.shape[0] - 1):
        if verbose and frame % 1000 == 0:
            print("# Frame {}".format(frame), end="\r")
        neighbor_change = covalently_bonded_oxygens[frame] != covalently_bonded_oxygens[frame + 1]
        if neighbor_change.any():
            jumping_protons, = neighbor_change.nonzero()
            for proton_index in jumping_protons:
                oxy_neighbor_before = covalently_bonded_oxygens[frame, proton_index]
                oxy_neighbor_after = covalently_bonded_oxygens[frame + 1, proton_index]
                oxygen_distance = atom_box.distance(
                    oxygen_trajectory[frame, oxy_neighbor_after],
                    oxygen_trajectory[frame, oxy_neighbor_before])
                jump_counter[(oxygen_distance - dmin) / (dmax - dmin) * bins] += 1
    print("")
    print("# Proton jump histogram:")
    for proton_index in range(jump_counter.size):
        print("{:10} {:10}".format(dmin + (dmax - dmin) / bins * (.5 + proton_index),
                                   jump_counter[proton_index]))

    print("# Jumps total: {:}".format(jump_counter.sum()))


def jump_histo2(dmin, dmax, bins, pbc, oxygen_trajectory, proton_trajectory,
                covalently_bonded_oxygens, verbose=False, nonorthogonal_box=False):
    """Counts proton jumps at different distances"""
    start_time = time.time()
    jump_probs = np.zeros(size=bins, dtype=float)

    if nonorthogonal_box:
        atom_box = AtomBoxMonoclin(pbc)
    else:
        atom_box = AtomBoxCubic(pbc)

    jump_counter = np.zeros(bins, int)

    for frame in range(oxygen_trajectory.shape[0] - 1):
        oxygen_distances = []
        if verbose and frame % 1000 == 0:
            print("# Frame {}".format(frame), end="\r")
        neighbor_change = covalently_bonded_oxygens[frame] != covalently_bonded_oxygens[frame + 1]
        if neighbor_change.any():
            jumping_protons, = neighbor_change.nonzero()
            for proton_index in jumping_protons:
                oxy_neighbor_before = covalently_bonded_oxygens[frame, proton_index]
                oxy_neighbor_after = covalently_bonded_oxygens[frame + 1, proton_index]
                oxygen_distance = atom_box.distance(
                    oxygen_trajectory[frame, oxy_neighbor_after],
                    oxygen_trajectory[frame, oxy_neighbor_before])
                oxygen_distances.append(oxygen_distance)
        histo, edges = np.histogram(distances, bins=bins, range=(dmin, dmax))
    print("")
    print("#Proton jump histogram:")
    for proton_index in range(jump_counter.size):
        print("{:10} {:10}".format(dmin + (dmax - dmin) / bins * (.5 + proton_index),
                                   jump_counter[proton_index]))

    print("#Jumps total: {:}".format(jump_counter.sum()))


def main(*args):
    parser = argparse.ArgumentParser(description="Jumpstats",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="trajectory")
    parser.add_argument("pbc", type=float, nargs="+",
                        help="Periodic boundaries. If 3 values are given, an orthorhombic cell is "
                             "assumed. If 9 values are given, the pbc vectors are constructed from "
                             "them")
    parser.add_argument("--dmin", type=float, default=2., help="Minimal value in Histogram")
    parser.add_argument("--dmax", type=float, default=3., help="Maximal value in Histogram")
    parser.add_argument("--bins", type=int, default=100, help="Maximal value in Histogram")
    parser.add_argument("--verbose", "-v", action="store_true", default="False", help="Verbosity")
    parser.add_argument("--mode", "-m", choices=["jumpprobs", "jumphisto"],
                        default="jumpprobs", help="Choose whether to calculate probability "
                                                  "histogram \
        or the histogram of proton jumps for different oxygen distances")
    args = parser.parse_args()

    if len(args.pbc) == 3:
        nonortho = False
    elif len(args.pbc) == 9:
        if args.verbose:
            print("#Got 9 pbc values. Assuming nonorthorhombic box")
        nonortho = True

    else:
        print("Wrong number of PBC arguments", file=sys.stderr)
        sys.exit(1)
    pbc = np.array(args.pbc)

    if args.verbose:
        print("# PBC used:\n#", pbc)

    trajectory = xyz_parser.load_atoms(args.filename)
    # trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True,
    #                                   verbose=args.verbose)
    BinDump.mark_acidic_protons(trajectory, pbc, nonortho=nonortho, verbose=args.verbose)
    oxygen_trajectory, proton_trajectory = cnpa.select_atoms(trajectory, "O", "AH")

    covalent_neighbors_filename = re.sub("\..{3}$", "", args.filename) + "_covevo.npy"
    if not os.path.exists(covalent_neighbors_filename):
        print("# Creating array of nearest oxygen neighbor over time for all protons")
        BinDump.npsave_covevo(covalent_neighbors_filename, oxygen_trajectory, proton_trajectory,
                              pbc, nonortho=nonortho, verbose=args.verbose)
    else:
        print("# Found array with nearest oxygen neighbor over time:", covalent_neighbors_filename)

    covalent_neighbors = np.load(covalent_neighbors_filename)

    if args.mode == "jumpprobs":
        jump_probs(oxygen_trajectory, proton_trajectory, covalent_neighbors, pbc, args.dmin,
                   args.dmax, args.bins, verbose=True,
                   nonortho=nonortho)
    else:
        jump_histo(args.dmin, args.dmax, args.bins, args.verbose, pbc, oxygen_trajectory,
                   proton_trajectory, covalent_neighbors, verbose=True, nonorthogonal_box=nonortho)


if __name__ == "__main__":
    main()
