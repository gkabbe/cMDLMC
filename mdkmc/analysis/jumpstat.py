#!/usr/bin/python -u
#encoding=utf-8

import sys
import numpy as np
import time
import argparse
import os
import re
import inspect

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from mdkmc.cython_exts.atoms import numpyatom as npa
from mdkmc.cython_exts.kMC import jumpstat_helper as jsh
from mdkmc.IO import BinDump

def determine_PO_pairs(O_frame, P_frame, pbc):
    P_neighbors = np.zeros(O_frame.shape[0], int)
    for i in xrange(O_frame.shape[0]):
        P_index = npa.nextNeighbor(O_frame[i], P_frame, pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors
    
def jump_histo(filename, dmin, dmax, bins, progress, pbc, frames, Os, Hs, covevo, verbose=False, nonortho=False):
        """Counts proton jumps at different distances"""
        start_time = time.time()
        
        if nonortho == True:
            h = np.array(pbc.reshape((3,3)).T, order="C")
            h_inv = np.array(np.linalg.inv(h), order="C")

        jumpcounter = np.zeros(bins, int)

        for frame in xrange(Os.shape[0]-1):
            if verbose == True and frame % 1000 == 0:
                print "#Frame {}".format(frame), "\r",
            neighborchange = covevo[frame]!=covevo[frame+1]
            if neighborchange.any():
                    jump_protons = neighborchange.nonzero()[0]
                    for i in jump_protons:
                            # pdb.set_trace()
                            O_before = covevo[frame, i]
                            O_after = covevo[frame+1, i]
                            if nonortho == True:
                                O_dist = npa.length_nonortho_bruteforce(Os[frame, O_after], Os[frame, O_before], h, h_inv)
                            else:
                                O_dist = npa.length(Os[frame, O_after], Os[frame, O_before], pbc)
                            jumpcounter[(O_dist-dmin)/(dmax-dmin)*bins] += 1
        print ""
        print "#Proton jump histogram:"
        for i in xrange(jumpcounter.size):
                print "{:10} {:10}".format(dmin+(dmax-dmin)/bins*(.5+i), jumpcounter[i])

        print "#Jumps total: {:}".format(jumpcounter.sum())


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
    parser.add_argument("--mode", "-m", choices=["jumpprobs", "jumphisto", "O_RDF"], default="jumpprobs", help = "Choose whether to calculate probability histogram \
        or the histogram of proton jumps for different oxygen distances")

    args = parser.parse_args()
    
    if len(args.pbc) == 3:
        nonortho = False
    elif len(args.pbc) == 9:
        if args.verbose == True:
            print "#Got 9 pbc values. Assuming nonorthorhombic box"
        nonortho = True
        
    else:
        print >> sys.stderr, "Wrong number of PBC arguments"
        sys.exit(1)
    pbc= np.array(args.pbc)

    if args.verbose == True:
        print "#PBC used:\n#", pbc

    trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True, verbose=args.verbose)
    BinDump.mark_acidic_protons(trajectory, pbc, nonortho=nonortho, verbose=args.verbose)
    atoms = npa.select_atoms(trajectory, "O", "AH")
    Os = atoms["O"]
    Hs = atoms["AH"]

    covevo_filename = re.sub("\..{3}$", "", args.filename)+"_covevo.npy"
    if not os.path.exists(covevo_filename):
        BinDump.npsave_covevo(covevo_filename, Os, Hs, pbc, nonortho=nonortho, verbose=args.verbose)

    covevo = np.load(covevo_filename)

    if args.mode == "jumpprobs":
        jsh.jump_probs(Os, Hs, covevo, pbc, args.dmin, args.dmax, args.bins, verbose=args.verbose, nonortho=nonortho)
    elif args.mode == "O_RDF":
        pass
    else:
        jump_histo(args.filename, args.dmin, args.dmax, args.bins, args.verbose, pbc, args.frames, Os, Hs, covevo, verbose=args.verbose, nonortho=nonortho)

if __name__ == "__main__":
    main()
