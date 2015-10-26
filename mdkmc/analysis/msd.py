#!/usr/bin/python -u

import numpy as np
import argparse
from math import ceil

from mdkmc.atoms import numpyatom as npa
from mdkmc.IO import BinDump
import ipdb


def calculate_msd(atom_traj, pbc, intervalnumber, intervallength):
    """Input: trajectory in numpy format (only proton positions), periodic boundaries, number of intervals, interval length
    Output: tuple of averaged msd and msd variance
    Method by Knuth https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""
    msd_mean = np.zeros((intervallength, 3), float)
    msd_var = np.zeros((intervallength, 3), float)
    totallength = atom_traj.shape[0]
    #move_mat= np.zeros((atom_traj.shape[0],atom_traj.shape[1],atom_traj.shape[2]))
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


def calculate_msd_multi_interval(atom_traj, pbc, subinterval_delay):
    """Uses intervals ranging between a length of 1 timestep up to the length of the whole trajectory.
    subinterval_delay: the time delay between two successive intervals of the same length"""
    total_length = atom_traj.shape[0]
    occurrence_counter = np.zeros((atom_traj.shape[0], atom_traj.shape[1], atom_traj.shape[2]))
    msd_mean = np.zeros((atom_traj.shape[0], atom_traj.shape[1], atom_traj.shape[2]))
    msd_var = np.zeros((atom_traj.shape[0], atom_traj.shape[1], atom_traj.shape[2]))
    delta = np.zeros((atom_traj.shape[0], atom_traj.shape[1], atom_traj.shape[2]))
    for interval_length in xrange(0, total_length - subinterval_delay, subinterval_delay):
        ind_helper = np.zeros((atom_traj.shape[0], atom_traj.shape[1], atom_traj.shape[2]))
        sqdist = npa.sqdist_np_multibox(atom_traj[interval_length: total_length, :, :],
                                        atom_traj[interval_length, :, :], pbc, axis_wise=True)
        ind_helper[:total_length - interval_length, :, :] = sqdist
        indices = np.where(ind_helper != 0)
        occurrence_counter[indices] += 1
        delta[indices] = sqdist[indices] - msd_mean[indices]
        msd_mean[indices] += delta[indices] / (occurrence_counter[indices] + 1)
        msd_var[indices] += delta[indices] * (sqdist[indices] - msd_mean[indices])
    msd_var /= (occurrence_counter - 1)
    msd_var2 = msd_var.mean(axis=1)
    msd_mean2 = msd_mean.mean(axis=1)
    return msd_mean2, msd_var2


def main(*args):
    parser = argparse.ArgumentParser(description="Determine Mean Square Displacement of MD trajectory")
    parser.add_argument("filename", help="Trajectory filename")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("intervalnumber", type=int, help="Number of intervals over which to average")
    parser.add_argument("intervallength", type=int, help="Interval length")
    parser.add_argument("mode", type=str, help="multi mode  refers to new multintervall analysis including change of boxes and single proton variance calculation (in multi mode choose intervalnumber equal to 1), single mode is gabriels standard MSD")
    parser.add_argument("--trajectory_cut", type=int, help="Restrict trajectory to specified number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    #parser.add_argument("--variance_all_H", type=bool, help="if true Variance not from intervalls, variance from every H atom ")
    parser.add_argument("--variance_all_H", action="store_true", help="if true Variance not from intervalls, variance from every H atom ")
    args = parser.parse_args()

    # settings for multi interval analysis
    subintervalldelay = 20
    resolution = 100
    # -------------------------------------


    intervalnumber = args.intervalnumber
    intervallength = args.intervallength
    variance_all_H = args.variance_all_H
    mode = args.mode
    pbc = np.array(args.pbc)
    trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True, remove_com=True, verbose=args.verbose)
    if args.trajectory_cut:
        if args.verbose:
            print "# Trajectory length: {} frames".format(trajectory.shape[0])
            print "# Trajectory will be cut from frame 0 to frame {}".format(args.trajectory_cut)
        trajectory = trajectory[:args.trajectory_cut]

    BinDump.mark_acidic_protons(trajectory, pbc, verbose=args.verbose)
    proton_traj = npa.select_atoms(trajectory, "AH")["AH"]
    if mode == 'single':
        msd_mean, msd_var = calculate_msd(proton_traj, pbc, intervalnumber, intervallength)
    if mode == 'multi':
        msd_mean, msd_var = calculate_msd_multi_interval(proton_traj[::resolution], pbc, intervalnumber,
                                                         int(intervallength / resolution), subintervalldelay,
                                                         variance_all_H)
        # use only first 60% of the calculated interval because of statistic
        msd_mean, msd_var = msd_mean[:int(msd_mean.shape[0]*0.6)], msd_var[:int(msd_var.shape[0]*0.6)]

    for mm, mv in zip(msd_mean.sum(axis=1), msd_var.sum(axis=1)):
        print mm, mv


if __name__ == "__main__":
    main()
