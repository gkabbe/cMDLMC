# coding=utf-8

#!/usr/bin/env python3
# -*- coding: utf-8

import argparse

import matplotlib.pylab as plt
from numba import jit
import numpy as np
import pint
from scipy.optimize import curve_fit

ureg = pint.UnitRegistry()


def load_interval_samples(filename, lines, intervals, columns, time_columns, *, verbose=False):
    data = np.loadtxt(filename, usecols=columns)
    time = np.loadtxt(filename, usecols=time_columns)

    if verbose:
        print(("# Shape of data:", data.shape))

    if intervals is not None and intervals < data.shape[0]:
        data = data[:intervals * lines]
    else:
        intervals = data.shape[0] / lines

    if verbose:
        print(("# Intervals:", intervals))

    data = data.flatten().reshape(intervals, lines, len(columns))

    return data


def load_intervals_intelligently(filename, var_prot_single, verbose=False):
    def get_settings_from_settings_output(lines):
        if verbose:
            print("Trying to determine KMC intervals from configuration file")
        settings = dict()
        for line in lines:
            if "print_freq" in line:
                settings["print_freq"] = int(line.split()[-1])
            elif "reset_freq" in line:
                settings["reset_freq"] = int(line.split()[-1])
            elif "sweeps" in line:
                settings["sweeps"] = int(line.split()[-1])
            elif len(settings) == 3:
                break
        try:
            interval_length = settings["reset_freq"] // settings["print_freq"]
            interval_number = settings["sweeps"] // settings["reset_freq"]
            return interval_length, interval_number
        except KeyError:
            return None

    def get_settings_from_average_at_the_end(lines):
        if verbose:
            print("Trying to load averaged output")
        interval_length = 0
        total_lines = 0
        count_interval = False
        count_total = True
        for line in lines:
            if "Averaged Results" in line:
                count_interval = True
                count_total = False
            if "Total time" in line:
                break
            if count_interval:
                interval_length += 1
            if count_total:
                total_lines += 1
        if interval_length != 0:
            # The two comments are counted as well, therefore we substract them here
            interval_length -= 2
            return interval_length, total_lines // interval_length
        else:
            return None

    def get_settings_from_msd_zeros(data, lines):
        if verbose:
            print("Trying to determine KMC intervals heuristically")
        intervals = np.where(data[:, 2] == 0)[0]
        interval_length = intervals[1]
        if "Averaged Results" in " ".join(lines):
            interval_number = intervals.size - 1
        else:
            interval_number = intervals.size
        return interval_length, interval_number

    if var_prot_single:
        data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    else:
        data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6))
    with open(filename, "r") as f:
        lines = f.readlines()
    try:
        interval_length, interval_number = get_settings_from_settings_output(lines)
    except TypeError:
        try:
            interval_length, interval_number = get_settings_from_average_at_the_end(lines)
        except TypeError:
            interval_length, interval_number = get_settings_from_msd_zeros(data, lines)

    data = data[:interval_number * interval_length]
    if var_prot_single:
        data = data.reshape((interval_number, interval_length, 10))
    else:
        data = data.reshape((interval_number, interval_length, 7))
    return data


def avg(filename, variance, var_prot_single, plot=False, verbose=False):

    data = load_intervals_intelligently(filename, var_prot_single)
    avg = data[:, :, 2:].mean(axis=0)
    time = data[0, :, 0:2]

    if variance:
        var = data[:, :, 2:].var(axis=0)
        return time, avg, var
    else:
        return time, avg


def get_observable_names(outfilename):
    with open(outfilename, "r") as f:
        comments = [line for line in list(f) if line[0] == "#"]

    for c in reversed(comments):
        if "Sweeps" in c:
            return c.split()


def bootstrap_msd(args):
    filename = args.file
    data = load_intervals_intelligently(filename)

    msd = data[:, :, 2:5].sum(axis=2)
    msd_mean = msd.mean(axis=0)
    msd_mean_diff = (np.roll(msd_mean, -1) - msd_mean)[:-1]
    msd_shift = np.roll(msd, -1, axis=1)
    differences = (msd_shift - msd)[:, :-1]
    ipdb.set_trace()


def get_slope(args):
    """Fit model of the form f(x) = m * x + y to each interval.
    Determine then mean and standard deviation of the resulting slopes."""
    def fit_func(x, m, y):
        return m * x + y

    msd_unit = args.length_unit**2 / args.time_unit

    filename, fit_startpoint = args.file, args.msd_fitstart
    data = load_intervals_intelligently(filename, args.var_prot_single, verbose=args.verbose)

    if len(data.shape) == 3:
        time = data[0, :, 1]
        y_avg = data[:, :, 2:5].sum(axis=-1).mean(axis=0)
        y_err = np.sqrt(data[:, :, 2:5].sum(axis=-1).var(axis=0))
        if args.average_first:
            params, cov_mat = curve_fit(fit_func, time[fit_startpoint:], y_avg[fit_startpoint:],
                                        sigma=y_err[fit_startpoint:], absolute_sigma=True)
            m, y0 = params
            m_err, y_0_err = np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])
        else:
            ms, y0s = [], []
            for interval in data:
                y = interval[:, 2:5].sum(axis=-1)
                params, cov_mat = curve_fit(fit_func, time[fit_startpoint:], y[fit_startpoint:])
                m, y0 = params
                ms.append(m)
                y0s.append(y0)
            ms = np.asfarray(ms)
            y0s = np.asfarray(y0s)
            y0 = y0s.mean()
            y0_err = np.std(y0s)
            m = ms.mean()
            m_err = np.std(ms)

    m, m_err = m * msd_unit, m_err * msd_unit

    if args.minimal:
        print(m.to(args.output_unit).m / 6, m_err.to(args.output_unit).m / 6)
    else:
        print("Slope:")
        print("({} ± {}) {}".format(m.to(args.output_unit).m, m_err.to(args.output_unit).m,
                                    args.output_unit))
        print("Diffusion coefficient:")
        print("({} ± {}) {}".format(m.to(args.output_unit).m / 6, m_err.to(args.output_unit).m / 6,
                                    args.output_unit))

    if args.plot:
        plt.errorbar(time, y_avg, y_err)
        plt.plot(time, time * m.m + y0)
        plt.plot(time, (time - time[args.msd_fitstart]) * (m.m + m_err.m) + y0 + time[
            args.msd_fitstart] * m.m, "g--")
        plt.plot(time, (time - time[args.msd_fitstart]) * (m.m - m_err.m) + y0 + time[
            args.msd_fitstart] * m.m, "g--")
        plt.vlines(time[args.msd_fitstart], *plt.ylim(), label="Fit start", colors="r")
        plt.legend()
        plt.show()


def average_kmc(args):
    kmc_out = args.file
    result = avg(kmc_out, variance=args.variance, var_prot_single=args.var_prot_single,
                 plot=args.plot, verbose=args.verbose)
    var_prot_single = args.var_prot_single
    comments = get_observable_names(kmc_out)

    if args.variance:
        print("# {:10} {:10}" + 6 * " {:12}" + 4 * " {:6}".format(comments[0], comments[1],
                                                                  comments[2],
                                                                  comments[2] + "_var", comments[3],
                                                                  comments[3] + "_var", comments[4],
                                                                  comments[4] + "_var", comments[5],
                                                                  comments[5] + "_var", comments[6],
                                                                  comments[6] + "_var"))
        format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msdvar[0]:12.4f} {msd[" \
                        "1]:12.4f} {msdvar[1]:12.4f} {msd[2]:12.4f} {msdvar[2]:12.4f} {" \
                        "autocorr:6.2f} {autocorrvar:6.2f} {jumps:6.2f} {jumpsvar:6.2f}"
        time, average, variance = result

        for i in range(average.shape[0]):
            print(format_string.format(t=time[i], msd=average[i, 0:3], msdvar=variance[i, 0:3],
                                       autocorr=average[i, 3], autocorrvar=variance[i, 3],
                                       jumps=average[i, 4], jumpsvar=variance[i, 4]))
    else:
        print("#", " ".join(["{:>10}", "{:>10}", 3 * "{:>12}", 2 * "{:>6}"]).format(*comments[1:]))
        if var_prot_single:
            format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msd[1]:12.4f} {msd[" \
                            "2]:12.4f} {msd_var[0]:12.4f} {msd_var[1]:12.4f} {msd_var[2]:12.4f}{" \
                            "autocorr:6.2f} {jumps:6.2f}"
        else:
            format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msd[1]:12.4f} {msd[" \
                            "2]:12.4f} {autocorr:6.2f} {jumps:6.2f}"
        time, average = result
        for i in range(average.shape[0]):
            if var_prot_single:
                print((format_string.format(t=time[i], msd=average[i, :3], msd_var=average[i, 3:6],
                                            autocorr=average[i, 6], jumps=average[i, 7])))
            else:
                print((format_string.format(t=time[i], msd=average[i, :3], autocorr=average[i, 3],
                                            jumps=average[i, 4])))


@jit(nopython=True)
def average_excess_proton_msd(kmc_data, interval_length, interval_delta, pbc, periodic=True):
    if kmc_data.shape[0] == interval_length:
        interval_number = 1
    else:
        interval_number = (kmc_data.shape[0] - interval_length) // interval_delta
    msds = np.zeros((interval_number, interval_length, 3))

    print("Averaging over", interval_number, "intervals")
    for i in range(interval_number):
        distance = np.zeros(3)
        for j in range(1, interval_length):
            diff = kmc_data[i * interval_delta + j] - kmc_data[i * interval_delta + j - 1]
            if periodic:
                for k in range(3):
                    while diff[k] > pbc[k] / 2:
                        diff[k] -= pbc[k]
                    while diff[k] < -pbc[k] / 2:
                        diff[k] += pbc[k]
            distance += diff
            msds[i, j] = distance**2

    return msds


def main(*args):
    parser = argparse.ArgumentParser(
        description="Average KMC output. Assuming time in first column")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--plot", "-p", action="store_true", help="Show plot")
    parser.add_argument("--time_unit", default="fs", type=ureg.parse_expression,
                        help="Time unit of MC output")
    parser.add_argument("--length_unit", type=ureg.parse_expression, default="angstrom",
                        help="Length unit of atom coordinates")
    parser.add_argument("--var_prot_single", action="store_true", default=False,
                        help="average variance, mode var_prot_single during LMC run necessary ")

    subparsers = parser.add_subparsers()

    parser_slope = subparsers.add_parser("slope", help="Only determine slope of MSD")
    parser_slope.add_argument("file", help="KMC output")
    parser_slope.add_argument("-a", "--average_first", action="store_true",
                              help="Average MSD first, then determine slope")
    parser_slope.add_argument("--msd-fitstart", "-s", type=int, default=0,
                              help="From which point in the MSD interval to start fitting")
    parser_slope.add_argument("--output_unit", "-u", default="angstrom**2/ps",
                              type=ureg.parse_expression, help="Unit of MSD result")
    parser_slope.add_argument("--minimal", "-m", action="store_true", help="Only output numbers")
    parser_slope.set_defaults(func=get_slope)

    parser_all = subparsers.add_parser("average", help="Average all columns from KMC output")
    parser_all.add_argument("file", help="KMC output")
    parser_all.add_argument("--variance", action="store_true", help="Also output variance")
    parser_all.set_defaults(func=average_kmc)

    parser_bootstrap = subparsers.add_parser("bootstrap", help="Create bootstrap samples of MSD")
    parser_bootstrap.set_defaults(func=bootstrap_msd)
    parser_bootstrap.add_argument("file", help="KMC output")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
