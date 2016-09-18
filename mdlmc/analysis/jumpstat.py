#!/usr/bin/env python3 -u

import sys
import numpy as np
import time
import argparse
import os
import re
import inspect
import ipdb

from mdlmc.cython_exts.atoms import numpyatom as cnpa
from mdlmc.cython_exts.LMC import jumpstat_helper as jsh
from mdlmc.IO import BinDump
from mdlmc.atoms import numpyatom as npa

def determine_phosphorus_oxygen_pairs(oxygen_frame, phosphorus_frame, pbc):
    P_neighbors = np.zeros(oxygen_frame.shape[0], int)
    for i in range(oxygen_frame.shape[0]):
        P_index = cnpa.next_neighbor(oxygen_frame[i], phosphorus_frame, pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors


def jump_histo(dmin, dmax, bins, pbc, frames, oxygen_trajectory, proton_trajectory,
               covalent_neighbors, verbose=False, nonorthogonal_box=False):
        """Counts proton jumps at different distances"""
        start_time = time.time()
        
        if nonorthogonal_box:
            h = np.array(pbc.reshape((3, 3)).T, order="C")
            h_inv = np.array(np.linalg.inv(h), order="C")

        jump_counter = np.zeros(bins, int)

        for frame in range(oxygen_trajectory.shape[0] - 1):
            if verbose and frame % 1000 == 0:
                print("# Frame {}".format(frame), "\r", end=' ')
            neighbor_change = covalent_neighbors[frame] != covalent_neighbors[frame + 1]
            if neighbor_change.any():
                    jumping_protons, = neighbor_change.nonzero()
                    for proton_index in jumping_protons:
                            oxy_neighbor_before = covalent_neighbors[frame, proton_index]
                            oxy_neighbor_after = covalent_neighbors[frame + 1, proton_index]
                            if nonorthogonal_box:
                                oxygen_distance = cnpa.length_nonortho_bruteforce(
                                    oxygen_trajectory[frame, oxy_neighbor_after],
                                    oxygen_trajectory[frame, oxy_neighbor_before], h, h_inv)
                            else:
                                oxygen_distance = cnpa.length(
                                    oxygen_trajectory[frame, oxy_neighbor_after],
                                    oxygen_trajectory[frame, oxy_neighbor_before],
                                    pbc)
                            jump_counter[(oxygen_distance - dmin) / (dmax - dmin) * bins] += 1
        print("")
        print("#Proton jump histogram:")
        for proton_index in range(jump_counter.size):
            print("{:10} {:10}".format(dmin + (dmax - dmin) / bins * (.5 + proton_index),
                                       jump_counter[proton_index]))

        print("#Jumps total: {:}".format(jump_counter.sum()))


def jump_probs2(Os, Hs, proton_neighbors, pbc, dmin, dmax, bins, nonortho, verbose=True):
    # edges_jumps, histo_jumps = np.histogram(bins=bins, range=(dmin, dmax))
    # _, histo_nojumps = np.histogram(bins=bins, range=(dmin, dmax))
    jumpprobs = np.zeros(bins, dtype=float)
    
    for oxygen_frame, next_oxygen_frame, covevo_frame, covevo_next_frame in zip(
                                    Os[:-1], Os[1:], proton_neighbors[:-1], proton_neighbors[1:]):
        neighborchange = covevo_frame != covevo_next_frame
        if neighborchange.any():
            donor_acceptor_index, = neighborchange.nonzero()
            donor_indices, acceptor_indices = covevo_frame[donor_acceptor_index], covevo_next_frame[donor_acceptor_index]
            donors, acceptors = oxygen_frame[donor_indices], oxygen_frame[acceptor_indices]
            distances = np.sqrt((npa.distance(donors, acceptors, pbc) ** 2).sum(axis=-1))
            histo_jumps, edges_jumps = np.histogram(distances, bins=bins, range=(dmin, dmax))


def main(*args):

    parser=argparse.ArgumentParser(description="Jumpstats", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="trajectory")
    parser.add_argument("pbc", type=float, nargs="+", help="Periodic boundaries. If 3 values are given, an orthorhombic cell is assumed.\
     If 9 values are given, the pbc vectors are constructed from them")
    #~ parser.add_argument("--pbc_vecs", type=float, nargs=9, help="PBC Vectors")
    parser.add_argument("--dmin", type=float, default=2., help = "Minimal value in Histogram")
    parser.add_argument("--dmax", type=float, default=3., help = "Maximal value in Histogram")
    parser.add_argument("--bins", type=int, default=100, help = "Maximal value in Histogram")
    parser.add_argument("--verbose", "-v", action = "store_true", default="False", help="Verbosity")
    parser.add_argument("--debug", "-d", action = "store_true", default="False", help="Debug?")
    parser.add_argument("--frames", "-f", type=int, default=0, help = "Number of frames in xyz file")
    parser.add_argument("--mode", "-m", choices=["jumpprobs", "jumphisto", "O_RDF"], 
        default="jumpprobs", help = "Choose whether to calculate probability histogram \
        or the histogram of proton jumps for different oxygen distances")
    args = parser.parse_args()

    if len(args.pbc) == 3:
        nonortho = False
    elif len(args.pbc) == 9:
        if args.verbose == True:
            print("#Got 9 pbc values. Assuming nonorthorhombic box")
        nonortho = True
        
    else:
        print("Wrong number of PBC arguments", file=sys.stderr)
        sys.exit(1)
    pbc= np.array(args.pbc)

    if args.verbose == True:
        print("# PBC used:\n#", pbc)

    trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True,
                                      verbose=args.verbose)
    BinDump.mark_acidic_protons(trajectory, pbc, nonortho=nonortho, verbose=args.verbose)
    Os, Hs = cnpa.select_atoms(trajectory, "O", "AH")

    covalent_neighbors_filename = re.sub("\..{3}$", "", args.filename)+"_covevo.npy"
    if not os.path.exists(covalent_neighbors_filename):
        print("# Creating array of nearest oxygen neighbor over time for all protons")
        BinDump.npsave_covevo(covalent_neighbors_filename, Os, Hs, pbc, nonortho=nonortho,
                              verbose=args.verbose)
    else:
        print("# Found array with nearest oxygen neighbor over time:", covalent_neighbors_filename)

    covalent_neighbors = np.load(covalent_neighbors_filename)

    if args.mode == "jumpprobs":
        jump_probs2(Os, Hs, covalent_neighbors, pbc, args.dmin, args.dmax, args.bins, verbose=True,
                    nonortho=nonortho)
    elif args.mode == "O_RDF":
        pass
    else:
        jump_histo(args.dmin, args.dmax, args.bins, args.verbose, pbc, args.frames, Os, Hs,
                   covalent_neighbors, verbose=True, nonorthogonal_box=nonortho)

if __name__ == "__main__":
    main()
