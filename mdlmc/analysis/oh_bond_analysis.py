#!/usr/bin/env python3

import argparse
import re
import os
from math import ceil
from collections import Counter
import logging
import sys

from mdlmc.analysis.proton_jump_statistics import determine_protonated_oxygens
import numpy as np

from mdlmc.IO import BinDump


logger = logging.getLogger(__name__)


def determine_hydronium_indices(covevo):
    logger.info("Determining array of hydronium indices")
    h3o_indices = np.zeros((covevo.shape[0], 1), dtype=int)
    for i in range(covevo.shape[0]):
        counter = Counter(covevo[i])
        occupied, _ = counter.most_common(1)[0]
        h3o_indices[i] = occupied
    return h3o_indices


def oh_bond_array_filename(trajectory_filename):
    oh_bond_filename = re.sub("\..{3}$", "", trajectory_filename) + "_cbo.npy"
    return oh_bond_filename


def load_oh_bonds(filename, pbc, *, verbose=False):
    oh_bond_filename = oh_bond_array_filename(filename)
    if not os.path.exists(oh_bond_filename):
        logger.info("OH-bond file not existing. Creating...")
        oxygens, hydrogens = BinDump.npload_atoms(filename, atomnames_list=["O", "H"],
                                                  return_tuple=True, verbose=verbose)
        oh_bonds = determine_protonated_oxygens(filename, pbc, verbose=verbose)
        np.save(oh_bond_filename, oh_bonds)
    logger.info("Loading Covevo File...")
    oh_bonds = np.load(oh_bond_filename)
    return oh_bonds


def autocorrelate(oh_vector: np.ndarray, interval_number: int, interval_length: int,
                  *, verbose=False):
    oh_bonds_avg = np.zeros((interval_number, interval_length), int)
    total_length = oh_vector.shape[0]

    if interval_number * interval_length <= total_length:
        interval_distance = interval_length
    else:
        diff = interval_number * interval_length - total_length
        interval_distance = interval_length - int(ceil(diff / float(interval_number - 1)))

    logger.info("Averaging over {} intervals of length {} with distance {} to each other".format(
        interval_number, interval_length, interval_distance)
    )

    for i in range(interval_number):
        print("{} / {}".format(i, interval_number), end="\r", file=sys.stderr)
        oh_bonds_avg[i] = (
        oh_vector[i * interval_distance:i * interval_distance + interval_length] == oh_vector[i * interval_distance]).sum(axis=1)
    print("", file=sys.stderr)

    result = oh_bonds_avg.mean(axis=0)
    return result


def main(*args):
    parser = argparse.ArgumentParser(
        description="Determine covalent OH bond autocorrelation function of MD trajectory")
    parser.add_argument("filename", help="Trajectory from which to load the oxygen topologies")
    parser.add_argument("interval_number", type=int,
                        help="Number of intervals over which to average")
    parser.add_argument("interval_length", type=int, help="Interval length")
    parser.add_argument("--pbc", nargs=3, type=float,
                        help="Periodic boundaries")
    parser.add_argument("--water", "-w", action="store_true",
                        help="Determine H3O+ autocorrelation in water")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    interval_number = args.interval_number
    interval_length = args.interval_length
    pbc = np.array(args.pbc)

    oh_bonds = load_oh_bonds(args.filename, pbc, verbose=args.verbose)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.water:
        oh_bonds = determine_hydronium_indices(oh_bonds)

    result = autocorrelate(oh_bonds, interval_number, interval_length)

    for val in result:
        print(val)


if __name__ == "__main__":
    main()
