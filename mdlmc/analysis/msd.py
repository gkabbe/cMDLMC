#!/usr/bin/env python3
# -*- coding: utf-8

import argparse
import logging
from math import ceil
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import pint
from numba import jit

from mdlmc.IO import xyz_parser
from mdlmc.atoms import numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from scipy.optimize import curve_fit

ureg = pint.UnitRegistry()
logger = logging.getLogger(__name__)


@jit(nopython=True)
def pbc_dist(dist, pbc):
    while (dist > pbc / 2).any():
        indices = np.where(dist > pbc / 2)
        dist[indices] -= pbc[indices[-1]]
    while (dist < -pbc / 2).any():
        indices = np.where(dist < -pbc / 2)
        dist[indices] += pbc[indices[-1]]
    return dist


def squared_distance(a1_pos, a2_pos, pbc, axis_wise=False):
    """Calculate squared distance using numpy vector operations"""
    dist = a1_pos - a2_pos
    dist = pbc_dist(dist, pbc)

    if axis_wise:
        return dist * dist
    else:
        return (dist * dist).sum(axis=1)


@jit(nopython=True)
def displacement(positions, pbc):
    displ = np.zeros(positions.shape)
    total_diff = np.zeros(3)

    for i in range(1, positions.shape[0]):
        for j in range(positions.shape[1]):
            diff = positions[i, j] - positions[i - 1, j]
            diff = pbc_dist(diff, pbc)
            total_diff += diff
            displ[i, j] = total_diff

    return displ


def calculate_msd(atom_traj, pbc, intervalnumber, intervallength):
    """
    Method by Knuth https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Parameters
    ----------
    atom_traj: array_like
    pbc: array_like
    intervalnumber: int
    intervallength: int

    Returns
    -------
    Tuple of averaged msd and msd variance
    """

    msd_mean = np.zeros((intervallength, 3), float)
    msd_var = np.zeros((intervallength, 3), float)
    totallength = atom_traj.shape[0]

    if intervalnumber * intervallength <= totallength:
        startdist = intervallength
    else:
        diff = intervalnumber * intervallength - totallength
        startdist = intervallength - int(ceil(diff / float(intervalnumber - 1)))

    logger.info("Total length of trajectory: {}".format(totallength))
    logger.info("Number of intervals: {}".format(intervalnumber))
    logger.info("Interval distance: {}".format(startdist))

    for i in range(intervalnumber):
        print("{: 8d} / {: 8d}".format(i, intervalnumber), end="\r", flush=True, file=sys.stderr)
        displ = displacement(atom_traj[i * startdist:i * startdist + intervallength, :, :], pbc)
        sqdist = (displ * displ).mean(axis=1)  # average over atom number
        delta = sqdist - msd_mean
        msd_mean += delta / (i + 1)
        msd_var += delta * (sqdist - msd_mean)
    print(file=sys.stderr)
    # average over particle number and number of intervals
    msd_var /= (intervalnumber - 1)

    return msd_mean, msd_var


def calculate_msd_multi_interval(atom_traj, pbc, subinterval_delay=1):
    """
    Uses intervals ranging between a length of 1 timestep up to the length of the whole trajectory.

    Parameters
    ----------
    atom_traj: array_like
        Atom trajectory
    pbc: array_like
        Periodic boundaries
    subinterval_delay:
        The time delay between two successive intervals of the same length

    Returns
    -------

    """
    total_length = atom_traj.shape[0]
    occurrence_counter = np.zeros(atom_traj.shape, dtype=int)
    interval_mask = np.zeros(atom_traj.shape, dtype=bool)
    msd_mean = np.zeros(atom_traj.shape)
    msd_var = np.zeros(atom_traj.shape)
    delta = np.zeros(atom_traj.shape)

    for starting_point in range(0, total_length - subinterval_delay, subinterval_delay):
        print(starting_point, end="\r", file=sys.stderr, flush=True)
        displ = displacement(atom_traj[starting_point:, :, :], pbc)
        sqdist = displ * displ
        sqdist.resize(atom_traj.shape)
        interval_mask[:] = 0
        interval_mask[:-starting_point] = 1
        occurrence_counter[interval_mask] += 1
        delta[interval_mask] = sqdist[interval_mask] - msd_mean[interval_mask]
        msd_mean[interval_mask] += delta[interval_mask] / (occurrence_counter[interval_mask] + 1)
        msd_var[interval_mask] += \
            delta[interval_mask] * (sqdist[interval_mask] - msd_mean[interval_mask])
    where_greater_one = occurrence_counter > 1
    where_equals_zero = occurrence_counter == 0
    occurrence_counter[where_greater_one] -= 1
    occurrence_counter[where_equals_zero] += 1
    msd_var /= occurrence_counter
    return msd_mean.mean(axis=1), msd_var.mean(axis=1)


def main(*args):
    parser = argparse.ArgumentParser(
        description="Determine Mean Square Displacement of MD trajectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser.add_argument("filename", help="Trajectory filename")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("timestep", type=ureg.parse_expression, help="MD timestep (e.g. 0.5fs)")
    parser.add_argument("atom", type=str, help="Atom name")
    parser.add_argument("--length_unit", type=ureg.parse_expression, default="angstrom",
                        help="Length unit of atom coordinates")
    parser.add_argument("--trajectory_cut", type=int,
                        help="Restrict trajectory to specified number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--fit_from", type=int, help="Fit MSD from specified data point")
    parser.add_argument("-u", "--output_unit", type=ureg.parse_expression,
                        default="angstrom**2/ps",
                        help="In which unit to output MSD and diffusion coefficient")
    parser.add_argument("--columns", "-c", type=int, nargs="+",
                        help="Which columns contain the position?")
    parser_fixed = subparsers.add_parser("single", help="Intervals with fixed size")
    parser_fixed.add_argument("intervalnumber", type=int,
                              help="Number of intervals over which to average")
    parser_fixed.add_argument("intervallength", type=int, help="Interval length")
    parser_multi = subparsers.add_parser("multi", help="Intervals with variable size")
    parser_multi.add_argument("--variance_all_H", action="store_true",
                              help="If set, determine variance over every proton trajectory")
    parser_multi.add_argument("--resolution", type=int, default=1, help="Use only every ith frame"
                                                                        " of the trajectory")
    parser_multi.add_argument("--subinterval_delay", type=int, default=1, help="Distance between"
                                                                               " intervals")
    parser_water = subparsers.add_parser("water", help="MSD for single excess proton")
    parser_water.add_argument("intervalnumber", type=int,
                              help="Number of intervals over which to average")
    parser_water.add_argument("intervallength", type=int, help="Interval length")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    pbc = np.array(args.pbc)
    atom_box = AtomBoxCubic(pbc)

    if type(args.timestep) != ureg.Quantity:
        raise ValueError("You forgot to assign a unit to timestep!")

    if args.columns:
        logger.debug("Reading text file {}".format(args.filename))
        trajectory = np.loadtxt(args.filename, usecols=args.columns)[:, None, :]
        logger.debug("Done reading")
    else:
        if os.path.splitext(args.filename)[1] == "npz":
            if args.atom == "AH":
                trajectory = xyz_parser.load_trajectory_from_npz(args.filename, verbose=args.verbose)
            else:
                trajectory, = xyz_parser.load_trajectory_from_npz(args.filename, args.atom,
                                                                  verbose=args.verbose)
        else:
            if args.atom == "AH":
                trajectory = xyz_parser.load_atoms(args.filename, verbose=args.verbose,
                                                    clip=args.trajectory_cut)
            else:
                trajectory, = xyz_parser.load_atoms(args.filename, args.atom, verbose=args.verbose,
                                                    clip=args.trajectory_cut)
        if args.atom == "AH":
            trajectory = npa.get_acidic_protons(trajectory, atom_box, verbose=args.verbose)

    if args.subparser_name == "multi":
        # -------------------------------------
        # settings for multi interval analysis
        subinterval_delay = args.subinterval_delay
        resolution = args.resolution
        # -------------------------------------
        msd_mean, msd_var = calculate_msd_multi_interval(trajectory[::resolution], pbc,
                                                         subinterval_delay)
        # use only first 60% of the calculated interval because of statistic
        msd_mean, msd_var = msd_mean[:int(msd_mean.shape[0] * 0.6)], \
                            msd_var[:int(msd_var.shape[0] * 0.6)]
        for i in range(msd_mean.shape[0]):
            print(i, msd_mean[i].sum(), msd_var[i].sum())

    else:
        intervalnumber = args.intervalnumber
        intervallength = args.intervallength

        msd_mean, msd_var = calculate_msd(trajectory, pbc, intervalnumber, intervallength)

    # sum over the three spatial coordinate axes
    msd_mean, msd_var = msd_mean.sum(axis=1), msd_var.sum(axis=1)

    if msd_mean.shape[0] > 50:
        step = msd_mean.shape[0] // 50
    else:
        step = 1

    if args.plot:
        output_time_unit = ureg.parse_expression(
            [k for k, v in args.output_unit.to_tuple()[1] if v == -1.0][0])
        output_length_unit = ureg.parse_expression(
            [k for k, v in args.output_unit.to_tuple()[1] if v == 2.0][0])
        time_w_unit = np.arange(0, msd_mean.shape[0], step) * args.timestep
        msd_w_unit = msd_mean[::step] * args.length_unit**2
        yerr_w_unit = np.sqrt(msd_var[::step]) * args.length_unit**2
        plt.errorbar(time_w_unit.to(output_time_unit).magnitude,
                     msd_w_unit.to(output_length_unit**2).magnitude,
                     yerr=yerr_w_unit.to(output_length_unit**2).magnitude)
        plt.xlabel("Time / " + output_time_unit.units.__str__())
        plt.ylabel(r"MSD / " + output_length_unit.units.__str__() + "**2")

    if args.fit_from:
        def fit_func(x, m, y):
            return m * x + y
        params, cov_mat = curve_fit(fit_func, np.arange(args.fit_from, msd_mean.shape[0]),
                                    msd_mean[args.fit_from:],
                                    sigma=np.sqrt(msd_var[args.fit_from:]), absolute_sigma=True)
        m, y_0 = params
        m_err, y_0_err = np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])

        m = m * args.length_unit**2 / args.timestep
        m_err = m_err * args.length_unit**2 / args.timestep

        print("\nSlope in {}:".format(args.output_unit))
        print("({:.2e} +/- {:.2e}) {}".format(m.to(args.output_unit).magnitude,
                                              m_err.to(args.output_unit).magnitude, args.output_unit))

        print("\nDiffusion coefficient in {}:".format(args.output_unit))
        print("({:.2e} +/- {:.2e}) {}".format(m.to(args.output_unit).magnitude / 6,
                                              m_err.to(args.output_unit).magnitude / 6,
                                              args.output_unit))

        if args.plot:
            fit = m * np.arange(msd_mean.shape[0]) * args.timestep + y_0 * args.length_unit**2
            t = np.arange(msd_mean.shape[0]) * args.timestep
            plt.plot(t.to(output_time_unit), fit.to(output_length_unit**2))
    if args.plot:
        plt.show()

if __name__ == "__main__":
    main()
