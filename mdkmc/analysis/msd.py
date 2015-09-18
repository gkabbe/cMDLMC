#!/usr/bin/python -u

import numpy as np
import argparse
from math import ceil

from mdkmc.atoms import numpyatom as npa
from mdkmc.IO import BinDump


# assumes that only those atoms that are to be evaluated, are in the trajectory
# shape[0]: number of frames
# shape[1]: number of atoms
# shape[2]: x,y,z pos
def calculate_msd(atom_traj, pbc, intervalnumber, intervallength):
    """Input: trajectory in numpy format (only proton positions), periodic boundaries, number of intervals, interval length
    Output: tuple of averaged msd and msd variance
    Method by Knuth https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""
    msd_mean = np.zeros((intervallength, 3), float)
    msd_var = np.zeros((intervallength, 3), float)
    totallength = atom_traj.shape[0]
    if intervalnumber * intervallength <= totallength:
        startdist = intervallength
    else:
        diff = intervalnumber * intervallength - totallength
        startdist = intervallength - int(ceil(diff / float(intervalnumber - 1)))

    for i in xrange(intervalnumber):
        sqdist = npa.sqdist_np(atom_traj[i * startdist:i * startdist + intervallength, :, :],
                               atom_traj[i * startdist, :, :], pbc, axis_wise=True)
        sqdist = sqdist.mean(axis=1)  # average over atom number
        delta = sqdist - msd_mean
        msd_mean += delta / (i + 1)
        msd_var += delta * (sqdist - msd_mean)
    # average over particle number and number of intervals
    msd_var /= (intervalnumber - 1)

    return msd_mean, msd_var


# ~ def bootstrap(msd_mean):

def main(*args):
    parser = argparse.ArgumentParser(description="Determine Mean Square Displacement of MD trajectory")
    parser.add_argument("filename", help="Trajectory filename")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("intervalnumber", type=int, help="Number of intervals over which to average")
    parser.add_argument("intervallength", type=int, help="Interval length")
    parser.add_argument("--trajectory_cut", type=int, help="Restrict trajectory to specified number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    intervalnumber = args.intervalnumber
    intervallength = args.intervallength
    pbc = np.array(args.pbc)
    trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True, remove_com=True, verbose=args.verbose)
    if args.trajectory_cut:
        if args.verbose:
            print "# Trajectory length: {} frames".format(trajectory.shape[0])
            print "# Trajectory will be cut from frame 0 to frame {}".format(args.trajectory_cut)
        trajectory = trajectory[:args.trajectory_cut]

    BinDump.mark_acidic_protons(trajectory, pbc, verbose=args.verbose)
    proton_traj = npa.select_atoms(trajectory, "AH")["AH"]

    msd_mean, msd_var = calculate_msd(proton_traj, pbc, intervalnumber, intervallength)

    for mm, mv in zip(msd_mean.sum(axis=1), msd_var.sum(axis=1)):
        print mm, mv


if __name__ == "__main__":
    main()
