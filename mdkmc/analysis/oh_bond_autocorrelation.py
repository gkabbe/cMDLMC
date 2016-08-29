#!/usr/bin/env python3

import argparse
import re
import os
from math import ceil

import numpy as np

from mdkmc.IO import BinDump


def main(*args):

    parser = argparse.ArgumentParser(
        description="Determine covalent bond autocorrelation function of MD trajectory")
    parser.add_argument("filename", help="Trajectory from which to load the oxygen topologies")
    parser.add_argument("intervalnumber", type=int,
                        help="Number of intervals over which to average")
    parser.add_argument("intervallength", type=int, help="Interval length")
    parser.add_argument("--pbc", nargs=3, type=float,
                        help="Trajectory from which to load the oxygen topologies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    intervalnumber = args.intervalnumber
    intervallength = args.intervallength
    pbc = np.array(args.pbc)
    covevo_filename = re.sub("\..{3}$", "", args.filename) + "_covevo.npy"
    if not os.path.exists(covevo_filename):
        if args.verbose == True:
            print("#Covevo file not existing. Creating...")
        oxygens, hydrogens = BinDump.npload_atoms(args.filename, atomnames_list=["O", "H"],
                                                  return_tuple=True, verbose=args.verbose)
        oxygens, hydrogens = np.array(oxygens["pos"], order="C"), np.array(
            hydrogens["pos"], order="C")
        BinDump.npsave_covevo(covevo_filename, oxygens, hydrogens, pbc, verbose=args.verbose)
    if args.verbose == True:
        print("#Loading Covevo File...")
    covevo = np.load(covevo_filename)

    covevo_avg = np.zeros((args.intervalnumber, args.intervallength), int)

    totallength = covevo.shape[0]
    if intervalnumber * intervallength <= totallength:
        startdist = intervallength
    else:
        diff = intervalnumber * intervallength - totallength
        startdist = intervallength - int(ceil(diff / float(intervalnumber - 1)))

    for i in range(intervalnumber):
        if args.verbose == True:
            print("# {} / {}".format(i, intervalnumber), end="\r")
        covevo_avg[i] = (covevo[i * startdist:i * startdist + intervallength]
                         == covevo[i * startdist]).sum(axis=1)
    print("")

    result = covevo_avg.sum(axis=0) / float(covevo_avg.shape[0])

    for i in range(result.shape[0]):
        print(result[i])

if __name__ == "__main__":
    main()
