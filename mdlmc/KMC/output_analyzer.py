import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mdlmc.misc.tools import argparse_compatible


logger = logging.getLogger(__name__)


@argparse_compatible
def lifetime_histogram(filename: str, bins: int, plot: bool = False, save: bool = False):
    last_number_of_jumps = 0
    last_jump_time = 0
    lifetimes = []

    print("# Calculating lifetimes")
    with open(filename, "r") as file_:
        for i, line in enumerate(file_):
            if i % 1000 == 0:
                print(i, end="\r", flush=True)
            if not line.lstrip().startswith("#"):
                line_split = line.split()
                time = float(line_split[1])
                jumps = int(line_split[5])
                if jumps != last_number_of_jumps:
                    lifetimes.append(time - last_jump_time)
                    last_jump_time = time
                    last_number_of_jumps = jumps

    histo, edges = np.histogram(lifetimes, bins=bins)
    centres = edges[:-1] + edges[1:]

    average = np.mean(lifetimes)
    stddev = np.std(lifetimes)

    if plot:
        plt.plot(centres, histo)
        plt.show()

    if save:
        np.savetxt("lifetime.txt", np.c_[centres, histo])

        with open("lifetime.txt", "a") as f:
            print("# Average:", average, file=f)
            print("# Standard Deviation:", stddev, file=f)
    else:
        for c, h in zip(centres, histo):
            print(c, h)
        print("# Average:", average)
        print("# Standard Deviation:", stddev)


@argparse_compatible
def lifetime_autocorrelation(filename: str, correlation_length: int, interval_distance: int):
    logger.debug("Loading data")
    data = np.genfromtxt(filename, usecols=(5,), dtype=int)

    correlated = np.zeros(correlation_length, dtype=int)
    counter = 0

    for i in range(0, data.size - correlation_length, interval_distance):
        correlated += data[i] == data[i: i + correlation_length]
        counter += 1

    correlated = correlated.astype(float) / counter

    for val in correlated:
        print(val)


def main():
    parser = argparse.ArgumentParser("Analyze hydronium lifetime",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()

    histo = subparsers.add_parser("histogram", help="Calculate lifetime histogram")
    histo.add_argument("filename", help="KMC output file")
    histo.add_argument("--bins", type=int, default=50, help="Number of bins")
    histo.add_argument("--plot", action="store_true", help="Plot result")
    histo.add_argument("--save", action="store_true", help="Save result to file")
    histo.set_defaults(func=lifetime_histogram)

    autocorr = subparsers.add_parser("autocorr", help="Calculate lifetime autocorrelation")
    autocorr.add_argument("filename", help="KMC output file")
    autocorr.add_argument("correlation_length", type=int, help="Correlation length")
    autocorr.add_argument("--interval_distance", "-d", default=20, type=int,
                          help="Temporal distance between intervals")
    autocorr.set_defaults(func=lifetime_autocorrelation)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    args.func(args)


if __name__ == "__main__":
    main()
