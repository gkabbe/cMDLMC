import argparse

import numpy as np
import matplotlib.pyplot as plt

from mdlmc.misc.tools import argparse_compatible


@argparse_compatible
def hydronium_lifetime(filename: str, bins: int, plot: bool = False):
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

    np.savetxt("lifetime.txt", np.c_[centres, histo])

    with open("lifetime.txt", "a") as f:
        print("# Average:", average, file=f)
        print("# Standard Deviation:", stddev, file=f)


def main():
    parser = argparse.ArgumentParser("Analyze hydronium lifetime",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="KMC output file")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins")
    parser.add_argument("--plot", action="store_true", help="Plot result")
    args = parser.parse_args()

    hydronium_lifetime(args)


if __name__ == "__main__":
    main()
