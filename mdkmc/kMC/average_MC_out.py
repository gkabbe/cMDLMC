#!/usr/bin/python
# -*- coding: utf-8

import numpy as np
import argparse
import ipdb, sys


def load_interval_samples(filename, lines, intervals, columns, time_columns):
    data=np.loadtxt(filename, usecols=columns)
    time = np.loadtxt(filename, usecols=time_columns)

    if verbose == True:
        print("#Shape of data:", data.shape)

    if intervals != None and intervals < data.shape[0]:
        data = data[:intervals * lines]
    else:
        intervals = data.shape[0]/lines

    if verbose == True:
        print("#Intervals:", intervals)

    data = data.flatten().reshape(intervals, lines, len(columns))
    
    return data
    
def load_intervals_intelligently(filename):
    data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6))
    intervals = np.where(data[:, 2] == 0)[0]
    interval_length = intervals[1]
    with open(filename, "r") as f:
        if "Averaged Results" in " ".join(f.readlines()):
            interval_number = intervals.size - 1
        else:
            interval_number = intervals.size
            
    data = data[:interval_number*interval_length]
    data = data.reshape((interval_number, interval_length, 7))
    return data


def avg(filename, variance, verbose=False):

    data = load_intervals_intelligently(filename)
    avg = data[:, :, 2:].mean(axis=0)
    
    time = data[0, :, 0:2]

    if variance == True:
        var = data[:,:,2:].var(axis=0)
        return time, avg, var
    else:
        return time, avg


def read_from_configfile(config_filename):
    data = dict()
    with open(config_filename, "r") as f:
        for line in f:
            if line.lstrip()[0] != "#":
                if len(line.split()) > 2:
                    data[line.split()[0]] = line.split()[1:]
                else:
                    data[line.split()[0]] = line.split()[1]
    return data
    

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
    import matplotlib.pylab as plt
    ipdb.set_trace()

    
def get_slope(args):
    """Fit model of the form f(x) = m * x + y to each interval.
    Determine then mean and standard deviation of the resulting slopes."""

    filename, fit_startpoint = args.file, args.msd_fitstart
    data = load_intervals_intelligently(filename)
    m, y = np.polyfit(data[0, fit_startpoint:, 1], data[:, fit_startpoint:, 2:5].sum(axis=2).T, 1)
    m_mean, m_stddev = m.mean(), np.sqrt(m.var())
    print("Slope in angström²/fs:")
    print(m_mean, m_stddev)
    print("Slope in pm²/ps:")
    print(m_mean*1e7, m_stddev*1e7)
    print("Diffusion coefficient in pm²/ps:")
    print(m_mean*1e7/6, m_stddev*1e7/6)

def average_kmc(args):
    kmc_out = args.file
    result = avg(kmc_out, args.variance, args.verbose)
    
    comments = get_observable_names(kmc_out)

    if args.variance:
        print "# {:10} {:10}"+6*" {:12}"+4*" {:6}".format(comments[0], comments[1], comments[2], comments[2]+"_var",  comments[3], comments[3]+"_var", comments[4], comments[4]+"_var", comments[5], comments[5]+"_var", comments[6], comments[6]+"_var")
        format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msdvar[0]:12.4f} {msd[1]:12.4f} {msdvar[1]:12.4f} {msd[2]:12.4f} {msdvar[2]:12.4f} {autocorr:6.2f} {autocorrvar:6.2f} {jumps:6.2f} {jumpsvar:6.2f}"
        time, average, variance = result
        
        for i in range(average.shape[0]):
            print format_string.format(t=time[i], msd=average[i,0:3], msdvar=variance[i,0:3], autocorr=average[i,3], autocorrvar=variance[i,3], jumps=average[i,4], jumpsvar=variance[i,4])
    else:
        print "#", " ".join(["{:>10}", "{:>10}", 3*"{:>12}", 2*"{:>6}"]).format(*comments[1:])
        format_string = "{t[0]:10.2f} {t[1]:10.2f} {msd[0]:12.4f} {msd[1]:12.4f} {msd[2]:12.4f} {autocorr:6.2f} {jumps:6.2f}"
        time, average = result
        for i in range(average.shape[0]):
            print(format_string.format(t=time[i], msd=average[i,:3], autocorr=average[i,3], jumps=average[i,4]))

def main(*args):

    parser=argparse.ArgumentParser(description="Average KMC output. Assuming time in first column")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    subparsers = parser.add_subparsers()
    parser_slope = subparsers.add_parser("slope", help="Only determine slope of MSD")
    parser_slope.add_argument("file", help="KMC output")
    parser_slope.add_argument("--msd-fitstart", "-s", type=int, help="From which point in the MSD interval to start fitting")
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
