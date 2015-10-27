#!/usr/bin/python -u
# -*- coding: utf-8

import numpy as np
import argparse
from math import ceil
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from mdkmc.atoms import numpyatom as npa
from mdkmc.IO import BinDump

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
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser.add_argument("filename", help="Trajectory filename")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("--trajectory_cut", type=int, help="Restrict trajectory to specified number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--fit_from", type=int, help="Fit MSD from specified data point")
    parser_fixed = subparsers.add_parser("single", help="Intervals with fixed size")
    parser_fixed.add_argument("intervalnumber", type=int, help="Number of intervals over which to average")
    parser_fixed.add_argument("intervallength", type=int, help="Interval length")
    parser_multi = subparsers.add_parser("multi", help="Intervals with variable size")
    parser_multi.add_argument("--variance_all_H", action="store_true", help="If set, determine variance over every proton trajectory")
    args = parser.parse_args()

    pbc = np.array(args.pbc)
    trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True, remove_com=True, verbose=args.verbose)
    if args.trajectory_cut:
        if args.verbose:
            print "# Trajectory length: {} frames".format(trajectory.shape[0])
            print "# Trajectory will be cut from frame 0 to frame {}".format(args.trajectory_cut)
        trajectory = trajectory[:args.trajectory_cut]

    BinDump.mark_acidic_protons(trajectory, pbc, verbose=args.verbose)
    proton_traj = npa.select_atoms(trajectory, "AH")["AH"]


    if args.subparser_name == "multi":
        # -------------------------------------
        # settings for multi interval analysis
        subinterval_delay = 20
        resolution = 100
        # -------------------------------------
        msd_mean, msd_var = calculate_msd_multi_interval(proton_traj[::resolution], pbc, subinterval_delay)
        # use only first 60% of the calculated interval because of statistic
        msd_mean, msd_var = msd_mean[:int(msd_mean.shape[0]*0.6)], msd_var[:int(msd_var.shape[0]*0.6)]

    else:
        intervalnumber = args.intervalnumber
        intervallength = args.intervallength
        msd_mean, msd_var = calculate_msd(proton_traj, pbc, intervalnumber, intervallength)

    msd_mean, msd_var = msd_mean.sum(axis=1), msd_var.sum(axis=1) # sum over the three spatial coordinate axes

    for mm, mv in zip():
        print mm, mv

    if msd_mean.shape[0] > 50:
        step = msd_mean.shape[0] / 50
    else:
        step = 1

    if args.plot:
        plt.errorbar(np.arange(0, msd_mean.shape[0], step), msd_mean[::step], yerr=np.sqrt(msd_var[::step]))

    if args.fit_from:
        def fit_func(x, m, y):
            return m*x + y
        params, cov_mat = curve_fit(fit_func, np.arange(args.fit_from, msd_mean.shape[0]), msd_mean[args.fit_from:],
                                    sigma=np.sqrt(msd_var[args.fit_from:]), absolute_sigma=True)
        m, y_0 = params
        m_err, y_0_err = np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])

        print "\nSlope in angström²/timestep:"
        print m, m_err
        print "\nSlope in pm²/timestep:"
        print m*1e4, m_err*1e4
        print "\nDiffusion coefficient in angström²/timestep:"
        print m/6, m_err/6
        print "\nDiffusion coefficient in pm²/timestep:"
        print m*1e4/6, m_err*1e4/6

        if args.plot:
            plt.plot(np.arange(msd_mean.shape[0]), m*np.arange(msd_mean.shape[0])+y_0)

    plt.show()

if __name__ == "__main__":
    main()
