#!/usr/bin/env python3

import argparse
import re
import os
from math import ceil
from collections import Counter

from mdlmc.analysis.proton_jump_statistics import determine_protonated_oxygens
import numpy as np

from mdlmc.IO import BinDump


def determine_hydronium_indices(covevo):
    h3o_indices = np.zeros((covevo.shape[0], 1), dtype=int)
    for i in range(covevo.shape[0]):
        counter = Counter(covevo[i])
        occupied, _ = counter.most_common(1)[0]
        h3o_indices[i] = occupied
    return h3o_indices


def main(*args):
    parser = argparse.ArgumentParser(
        description="Determine covalent bond autocorrelation function of MD trajectory")
    parser.add_argument("filename", help="Trajectory from which to load the oxygen topologies")
    parser.add_argument("intervalnumber", type=int,
                        help="Number of intervals over which to average")
    parser.add_argument("intervallength", type=int, help="Interval length")
    parser.add_argument("--pbc", nargs=3, type=float,
                        help="Periodic boundaries")
    parser.add_argument("--water", "-w", action="store_true",
                        help="Determine H3O+ autocorrelation in water")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    intervalnumber = args.intervalnumber
    intervallength = args.intervallength
    pbc = np.array(args.pbc)
    covevo_filename = re.sub("\..{3}$", "", args.filename) + "_cbo.npy"
    if not os.path.exists(covevo_filename):
        if args.verbose:
            print("# Covevo file not existing. Creating...")
        oxygens, hydrogens = BinDump.npload_atoms(args.filename, atomnames_list=["O", "H"],
                                                  return_tuple=True, verbose=args.verbose)
        oxygens, hydrogens = np.array(oxygens["pos"], order="C"), np.array(
            hydrogens["pos"], order="C")
        covevo = determine_protonated_oxygens(covevo_filename, oxygens, hydrogens, pbc,
                                              verbose=args.verbose)
        np.save(covevo_filename, covevo)
    if args.verbose:
        print("# Loading Covevo File...")
    covevo = np.load(covevo_filename)

    if args.water:
        if args.verbose:
            print("# Determining array of hydronium indices")
        covevo = determine_hydronium_indices(covevo)

    covevo_avg = np.zeros((args.intervalnumber, args.intervallength), int)

    totallength = covevo.shape[0]
    if intervalnumber * intervallength <= totallength:
        startdist = intervallength
    else:
        diff = intervalnumber * intervallength - totallength
        startdist = intervallength - int(ceil(diff / float(intervalnumber - 1)))

    if args.verbose:
        print("# Averaging over", intervalnumber, "intervals of length", intervallength,
              "with distance", startdist, "to each other")

    for i in range(intervalnumber):
        if args.verbose:
            print("# {} / {}".format(i, intervalnumber), end="\r")
        covevo_avg[i] = (
        covevo[i * startdist:i * startdist + intervallength] == covevo[i * startdist]).sum(axis=1)
    print("")

    result = covevo_avg.mean(axis=0)

    for i in range(result.shape[0]):
        print(result[i])


if __name__ == "__main__":
    main()
