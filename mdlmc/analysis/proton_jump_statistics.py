#!/usr/bin/env python3 -u

import time
import argparse
import os
import re
from collections import Counter
import ipdb

import numpy as np
import matplotlib.pylab as plt

from mdlmc.IO import xyz_parser
from mdlmc.atoms import numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxMonoclinic, AtomBoxCubic
from mdlmc.misc.tools import argparse_compatible


def determine_phosphorus_oxygen_pairs(oxygen_frame, phosphorus_frame, atom_box):
    P_neighbors = np.zeros(oxygen_frame.shape[0], int)
    for i in range(oxygen_frame.shape[0]):
        P_index = atom_box.next_neighbor(oxygen_frame[i], phosphorus_frame)[0]
        P_neighbors[i] = P_index
    return P_neighbors


def find_oxonium_ions(covalently_bonded_oxygen_frame):
    oxonium_ions = []
    counter = Counter(covalently_bonded_oxygen_frame)
    for k, v in counter.items():
        if v == 3:
            oxonium_ions.append(k)
    return np.array(oxonium_ions)


def determine_protonated_oxygens(trajectory, pbc, *, nonorthorhombic_box=False,
                                 verbose=False):
    """Saves for each proton the index of its closest oxygen neighbor for every time step"""

    if nonorthorhombic_box:
        if verbose:
            print("# Nonorthogonal periodic boundary conditions")
        atom_box = AtomBoxMonoclinic(pbc)
    else:
        if verbose:
            print("# Cubic periodic boundary conditions")
        atom_box = AtomBoxCubic(pbc)

    print("# Determining acidic protons...")
    proton_indices = npa.get_acidic_proton_indices(trajectory[0], atom_box, verbose=verbose)

    proton_trajectory = np.array(trajectory["pos"][:, proton_indices])
    oxygen_trajectory, = npa.select_atoms(trajectory, "O")

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
                  end='\r', flush=True)
    print("")

    return covalently_bonded_oxygens


@argparse_compatible
def proton_jump_probability_at_oxygen_distance(filename, dmin, dmax, bins, pbc, *, verbose=False,
                                               nonorthogonal_box=False, water=False):
    """Determine the probability of a proton jump for a given oxygen distance.
    For each frame a distance histogram of oxygen pairs with one oxygen bonded to a proton is
    determined. Then, a distance histogram of the oxygen pairs between which a proton jump occurs at
    the next time step, is determined.
    Dividing the first with the latter results in a probability for a proton jump.
    In the case of water, a proton jump can occur between an oxonium ion and a neutral water ion."""

    pbc = np.array(pbc)
    if len(pbc) == 3:
        atom_box = AtomBoxCubic(pbc)
        nonorthorhombic_box = False
    elif len(pbc) == 9:
        if args.verbose:
            print("# Got 9 PBC values. Assuming nonorthorhombic box")
        atom_box = AtomBoxMonoclinic(pbc)
        nonorthorhombic_box = True
    else:
        raise ValueError("Wrong number of PBC arguments")

    if verbose:
        print("# Periodic box lengths used:")
        print("#", pbc)

    atoms = xyz_parser.load_atoms(filename, verbose=verbose)

    protonated_oxygens_filename = re.sub("\..{3}$", "", filename) + "_cbo.npy"
    if not os.path.exists(protonated_oxygens_filename):
        print("# Creating array of nearest oxygen neighbor over time for all protons")
        protonated_oxygens = determine_protonated_oxygens(atoms, pbc,
                                                          nonorthorhombic_box=nonorthorhombic_box,
                                                          verbose=verbose)
        np.save(protonated_oxygens_filename, protonated_oxygens)
    else:
        print("# Found array with nearest oxygen neighbor over time:", protonated_oxygens_filename)
        protonated_oxygens = np.load(protonated_oxygens_filename)

    jump_probabilities = np.zeros(bins, float)
    oxygen_distance_histo = np.zeros(bins, int)
    oxygen_trajectory, = npa.select_atoms(atoms, "O")
    counter = np.zeros(bins, int)

    for frame in range(oxygen_trajectory.shape[0] - 1):
        oxygen_distances_at_jump = []
        if verbose and frame % 1000 == 0:
            print("# Frame {}".format(frame), end="\r", flush=True)
        neighbor_change = protonated_oxygens[frame] != protonated_oxygens[frame + 1]
        if neighbor_change.any():
            jumping_protons, = neighbor_change.nonzero()
            for proton_index in jumping_protons:
                oxy_neighbor_before = protonated_oxygens[frame, proton_index]
                oxy_neighbor_after = protonated_oxygens[frame + 1, proton_index]
                oxygen_distance = atom_box.length(oxygen_trajectory[frame, oxy_neighbor_after],
                                                  oxygen_trajectory[frame, oxy_neighbor_before])
                oxygen_distances_at_jump.append(oxygen_distance[0])
        histo_jump, edges = np.histogram(oxygen_distances_at_jump, bins=bins, range=(dmin, dmax))
        if water:
            occupied_oxygen_indices = find_oxonium_ions(protonated_oxygens[frame])
        else:
            occupied_oxygen_indices = protonated_oxygens[frame]

        occ_mask = np.zeros(oxygen_trajectory.shape[1], bool)
        occ_mask[occupied_oxygen_indices] = 1
        all_to_all = atom_box.length_all_to_all(oxygen_trajectory[frame, occ_mask],
                                                oxygen_trajectory[frame, ~occ_mask])
        histo_ox, edges = np.histogram(all_to_all, bins=bins, range=(dmin, dmax))
        mask = histo_ox != 0
        counter += mask
        jump_probabilities[mask] += (np.asfarray(histo_jump[mask]) / histo_ox[mask])
        oxygen_distance_histo += histo_jump
    jump_probabilities /= counter

    ox_dists = (edges[:-1] + edges[1:]) / 2

    print("")
    print("# Proton jump histogram:")
    print("# Oxygen distance, jump probability, oxygen distance histogram")
    print("#", 60 * "-")
    for ox_dist, jump_prob, oxy_histo in zip(ox_dists, jump_probabilities, oxygen_distance_histo):
        print("  {:>15.8f}  {:>16.8f}  {:>25}".format(ox_dist, jump_prob, oxy_histo))


@argparse_compatible
def oxygen_and_proton_motion_during_a_jump(filename, pbc, time_window, *,
                                           nonorthorhombic_box=False, water=False, plot=False,
                                           verbose=False):
    atoms = xyz_parser.load_atoms(filename)

    protonated_oxygens_filename = re.sub("\..{3}$", "", filename) + "_cbo.npy"
    if not os.path.exists(protonated_oxygens_filename):
        print("# Creating array of nearest oxygen neighbor over time for all protons")
        protonated_oxygens = determine_protonated_oxygens(atoms, pbc,
                                                          nonorthorhombic_box=nonorthorhombic_box,
                                                          verbose=verbose)
        np.save(protonated_oxygens_filename, protonated_oxygens)
    else:
        print("# Found array with nearest oxygen neighbor over time:", protonated_oxygens_filename)
        protonated_oxygens = np.load(protonated_oxygens_filename)

    oxygen_distances = []
    proton_acceptor_distances = []
    donor_proton_distances = []

    if nonorthorhombic_box:
        atombox = AtomBoxMonoclinic(pbc)
    else:
        atombox = AtomBoxCubic(pbc)

    oxygens, = npa.select_atoms(atoms, "O")
    protons = npa.get_acidic_protons(atoms, atombox)

    for frame, (protonated_oxygens_before, protonated_oxygens_after) in enumerate(
        zip(protonated_oxygens[time_window // 2 - 1: -time_window // 2 - 1],
            protonated_oxygens[time_window // 2: -time_window // 2]), start=time_window // 2):
        protonation_change = protonated_oxygens_before != protonated_oxygens_after
        if protonation_change.any():
            proton_indices, = np.where(protonation_change)
            donor_indices = protonated_oxygens_before[proton_indices]
            acceptor_indices = protonated_oxygens_after[proton_indices]
            oxygen_dists = atombox.length(
                oxygens[frame - time_window // 2: frame + time_window // 2, donor_indices],
                oxygens[frame - time_window // 2: frame + time_window // 2, acceptor_indices])
            donor_proton_dists = atombox.length(
                oxygens[frame - time_window // 2: frame + time_window // 2, donor_indices],
                protons[frame - time_window // 2: frame + time_window // 2, proton_indices])
            proton_acceptor_dists = atombox.length(
                protons[frame - time_window // 2: frame + time_window // 2, proton_indices],
                oxygens[frame - time_window // 2: frame + time_window // 2, acceptor_indices])

            for dists in oxygen_dists.T:
                oxygen_distances.append(dists)
            for dists in donor_proton_dists.T:
                donor_proton_distances.append(dists)
            for dists in proton_acceptor_dists.T:
                proton_acceptor_distances.append(dists)

        if frame % 1000 == 0:
            print(frame, end="\r")

    proton_acceptor_distances = np.array(proton_acceptor_distances).mean(axis=0)
    proton_acceptor_distances_var = np.array(proton_acceptor_distances).var(axis=0)
    donor_proton_distances = np.array(donor_proton_distances).mean(axis=0)
    donor_proton_distances_var = np.array(donor_proton_distances).var(axis=0)
    oxygen_distances = np.array(oxygen_distances).mean(axis=0)
    oxygen_distances_var = np.array(oxygen_distances).var(axis=0)

    if plot:
        plt.plot(proton_acceptor_distances)
        plt.fill_between(np.arange(proton_acceptor_distances.size),
                         proton_acceptor_distances - np.sqrt(proton_acceptor_distances_var),
                         proton_acceptor_distances + np.sqrt(proton_acceptor_distances_var),
                         alpha=0.5)
        plt.title("Proton - Acceptor Distances")
        plt.show()

        plt.plot(donor_proton_distances)
        plt.fill_between(np.arange(donor_proton_distances.size),
                         donor_proton_distances - np.sqrt(donor_proton_distances_var),
                         donor_proton_distances + np.sqrt(donor_proton_distances_var),
                         alpha=0.5)
        plt.title("Proton - Donor Distances")
        plt.show()

        plt.plot(oxygen_distances)
        plt.fill_between(np.arange(oxygen_distances.size),
                         oxygen_distances - np.sqrt(oxygen_distances_var),
                         oxygen_distances + np.sqrt(oxygen_distances_var),
                         alpha=0.5)
        plt.title("Proton - Acceptor Distances")
        plt.show()


def main(*args):

    parser = argparse.ArgumentParser(description="Proton jump statistics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbosity")
    parser.add_argument("--water", "-w", action="store_true", default=False,
                        help="Set to true if you analyze a water trajectory")

    subparsers = parser.add_subparsers(dest="subparser_name")

    parser_jumpprob = subparsers.add_parser("jumpprobs")
    parser_jumpprob.add_argument("filename", help="trajectory")
    parser_jumpprob.add_argument("pbc", type=float, nargs="+",
                                 help="Periodic boundaries. If 3 values are given, an orthorhombic "
                                      "cell is assumed. If 9 values are given, the pbc vectors are "
                                      "constructed from them")
    parser_jumpprob.add_argument("--dmin", type=float, default=2., help="Minimal value in Histogram")
    parser_jumpprob.add_argument("--dmax", type=float, default=3., help="Maximal value in Histogram")
    parser_jumpprob.add_argument("--bins", type=int, default=100, help="Maximal value in Histogram")
    parser_jumpprob.set_defaults(func=proton_jump_probability_at_oxygen_distance)

    parser_jumpmotion = subparsers.add_parser("jumpmotion")
    parser_jumpmotion.add_argument("filename", help="trajectory")
    parser_jumpmotion.add_argument("pbc", type=float, nargs="+",
                                   help="Periodic boundaries. If 3 values are given, an orthorhombic"
                                        " cell is assumed. If 9 values are given, the pbc vectors "
                                        "are constructed from them")
    parser_jumpmotion.add_argument("--time_window", "-t", type=int, default=200,
                                   help="Length of the time window over which the proton and oxygen "
                                                                   "motion are averaged")
    parser_jumpmotion.add_argument("--plot", action="store_true", help="Plot result")
    parser_jumpmotion.set_defaults(func=oxygen_and_proton_motion_during_a_jump)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
