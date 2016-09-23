#!/usr/bin/env python3 -u

import sys
import time
import argparse
import os
import re

import numpy as np
import ipdb

from mdlmc.IO import xyz_parser
from mdlmc.atoms import numpyatom as npa
from mdlmc.cython_exts.LMC.LMCHelper import AtomBoxMonoclinic, AtomBoxCubic


def determine_phosphorus_oxygen_pairs(oxygen_frame, phosphorus_frame, atom_box):
    P_neighbors = np.zeros(oxygen_frame.shape[0], int)
    for i in range(oxygen_frame.shape[0]):
        P_index = atom_box.next_neighbor(oxygen_frame[i], phosphorus_frame)[0]
        P_neighbors[i] = P_index
    return P_neighbors


def determine_covalently_bonded_oxygens(trajectory, pbc, *, nonorthorhombic_box=False,
                                        verbose=False):
    """Saves for each proton at every time step the index of its closest oxygen neighbor"""

    if nonorthorhombic_box:
        if verbose:
            print("# Nonorthogonal periodic boundary conditions")
        atom_box = AtomBoxMonoclinic(pbc)
    else:
        if verbose:
            print("# Cubic periodic boundary conditions")
        atom_box = AtomBoxCubic(pbc)

    print("# Determining acidic protons...")
    proton_indices = npa.get_acidic_protons(trajectory[0], atom_box, verbose=verbose)

    proton_trajectory = np.array(trajectory["pos"][:, proton_indices])
    oxygen_trajectory = npa.select_atoms(trajectory, "O")

    print("# Determining covalently bonded oxygens over time..")
    start_time = time.time()
    covalently_bonded_oxygens = np.zeros((proton_trajectory.shape[0], proton_trajectory.shape[1]),
                                         dtype=int)

    for i in range(covalently_bonded_oxygens.shape[0]):
        for j in range(covalently_bonded_oxygens.shape[1]):
            covalently_bonded_oxygens[i, j], _ = \
                atom_box.next_neighbor(proton_trajectory[i, j], oxygen_trajectory[i])
        if verbose and i % 100 == 0:
            print("# Frame {: 6d} ({:5.0f} fps)".format(i, float(i) / (time.time() - start_time)),
                  end='\r')
    print("")

    return covalently_bonded_oxygens


def oxygen_distance_at_proton_jump(dmin, dmax, bins, pbc, atoms, covalently_bonded_oxygens,
                                   verbose=False, nonorthorhombic_box=False):
    """Determines the histogram of oxygen distances at which proton jumps occur"""
    start_time = time.time()

    if nonorthorhombic_box:
        atom_box = AtomBoxMonoclinic(pbc)
    else:
        atom_box = AtomBoxCubic(pbc)

    jump_counter = np.zeros(bins, int)

    oxygen_trajectory = npa.select_atoms(atoms, "O")

    for frame in range(oxygen_trajectory.shape[0] - 1):
        oxygen_distances = []
        if verbose and frame % 1000 == 0:
            print("# Frame {}".format(frame), end="\r")
        # Detect jumps by checking, when the next oxygen neighbor of a proton changes
        neighbor_change = covalently_bonded_oxygens[frame] != covalently_bonded_oxygens[frame + 1]
        if neighbor_change.any():
            jumping_protons, = neighbor_change.nonzero()
            for proton_index in jumping_protons:
                oxy_neighbor_before = covalently_bonded_oxygens[frame, proton_index]
                oxy_neighbor_after = covalently_bonded_oxygens[frame + 1, proton_index]
                oxygen_distance = atom_box.distance(oxygen_trajectory[frame, oxy_neighbor_after],
                                                    oxygen_trajectory[frame, oxy_neighbor_before])
                oxygen_distances.append(oxygen_distance)
        histo, edges = np.histogram(oxygen_distances, bins=bins, range=(dmin, dmax))
        jump_counter += histo
    print("")
    print("# Proton jump histogram:")
    for proton_index in range(jump_counter.size):
        print("{:10} {:10}".format(dmin + (dmax - dmin) / bins * (.5 + proton_index),
                                   jump_counter[proton_index]))

    print("# Jumps total: {:}".format(jump_counter.sum()))


def proton_jump_probability_at_oxygen_distance(dmin, dmax, bins, pbc, atoms,
                                               covalently_bonded_oxygens, *, verbose=False,
                                               nonorthogonal_box=False):
    """Determines the probability of a proton jump for a given oxygen distance"""
    start_time = time.time()
    jump_probs = np.zeros(bins, dtype=float)

    if nonorthogonal_box:
        atom_box = AtomBoxMonoclinic(pbc)
    else:
        atom_box = AtomBoxCubic(pbc)

    jump_probabilities = np.zeros(bins, float)
    oxygen_trajectory = npa.select_atoms(atoms, "O")
    counter = np.zeros(bins, int)

    for frame in range(oxygen_trajectory.shape[0] - 1):
        oxygen_distances_at_jump = []
        if verbose and frame % 1000 == 0:
            print("# Frame {}".format(frame), end="\r", flush=True)
        neighbor_change = covalently_bonded_oxygens[frame] != covalently_bonded_oxygens[frame + 1]
        if neighbor_change.any():
            jumping_protons, = neighbor_change.nonzero()
            for proton_index in jumping_protons:
                oxy_neighbor_before = covalently_bonded_oxygens[frame, proton_index]
                oxy_neighbor_after = covalently_bonded_oxygens[frame + 1, proton_index]
                oxygen_distance = atom_box.distance(oxygen_trajectory[frame, oxy_neighbor_after], oxygen_trajectory[frame, oxy_neighbor_before])
                oxygen_distances_at_jump.append(oxygen_distance)

        histo_jump, edges = np.histogram(oxygen_distances_at_jump, bins=bins, range=(dmin, dmax))
        histo_ox, edges = np.histogram(
            atom_box.distance_all_to_all(oxygen_trajectory[frame], oxygen_trajectory[frame]),
            bins=bins, range=(dmin, dmax))
        mask = histo_ox != 0
        counter += mask
        jump_probs[mask] += np.asfarray(histo_jump[mask]) / histo_ox[mask]
    jump_probs /= counter

    ox_dists = (edges[:-1] + edges[1:]) / 2

    print("")
    print("# Proton jump histogram:")
    for ox_dist, jump_prob in zip(ox_dists, jump_probs):
        print(ox_dist, jump_prob)


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
        nonorthorhombic_box = False
    elif len(args.pbc) == 9:
        if args.verbose:
            print("#Got 9 pbc values. Assuming nonorthorhombic box")
        nonorthorhombic_box = True

    else:
        print("Wrong number of PBC arguments", file=sys.stderr)
        sys.exit(1)
    pbc = np.array(args.pbc)

    if args.verbose:
        print("# PBC used:\n#", pbc)

    atoms = xyz_parser.load_atoms(args.filename, verbose=args.verbose)

    covalent_neighbors_filename = re.sub("\..{3}$", "", args.filename) + "_cbo.npy"
    if not os.path.exists(covalent_neighbors_filename):
        print("# Creating array of nearest oxygen neighbor over time for all protons")
        covalent_neighbors = determine_covalently_bonded_oxygens(atoms, pbc,
                                                                 nonorthorhombic_box=nonorthorhombic_box,
                                                                 verbose=args.verbose)
        np.save(covalent_neighbors_filename, covalent_neighbors)
    else:
        print("# Found array with nearest oxygen neighbor over time:", covalent_neighbors_filename)
        covalent_neighbors = np.load(covalent_neighbors_filename)

    if args.mode == "jumpprobs":
        proton_jump_probability_at_oxygen_distance(args.dmin, args.dmax, args.bins, pbc, atoms,
                                                   covalent_neighbors, verbose=True,
                                                   nonorthogonal_box=nonorthorhombic_box)
    else:
        oxygen_distance_at_proton_jump(args.dmin, args.dmax, args.bins, pbc, atoms,
                                       covalent_neighbors, verbose=True,
                                       nonorthorhombic_box=nonorthorhombic_box)


if __name__ == "__main__":
    main()
