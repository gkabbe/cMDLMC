#!/usr/bin/python -u

import numpy as np
import argparse
import re
import os
from math import ceil

from mdkmc.IO import BinDump

def main(*args):
    
    parser=argparse.ArgumentParser(description="Determine covalent bond autocorrelation function of MD trajectory")
    parser.add_argument("filename", help="Trajectory from which to load the oxygen topologies")
    parser.add_argument("intervalnumber", type=int, help="Number of intervals over which to average")
    parser.add_argument("intervallength", type=int, help="Interval length")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    args = parser.parse_args()    
    
    intervalnumber = args.intervalnumber
    intervallength = args.intervallength
    
    covevo_filename = re.sub("\..{3}$", "", args.filename)+"_covevo.npy"
    if not os.path.exists(covevo_filename):
        if args.verbose == True:
            print "#Covevo file not existing. Creating..."
        BinDump.npsave_covevo(covevo_filename, Os, Hs, pbc, verbose=args.verbose)
    if args.verbose == True:
        print "#Loading Covevo File..."
    covevo = np.load(covevo_filename)
    
    covevo_avg = np.zeros((args.intervalnumber, args.intervallength), int)
    
    totallength = covevo.shape[0]
    if intervalnumber*intervallength <= totallength:
        startdist=intervallength
    else:
        diff = intervalnumber*intervallength-totallength
        startdist = intervallength-int(ceil(diff/float(intervalnumber-1)))

    for i in xrange(intervalnumber):
        if args.verbose == True:
            print "# {} / {}".format(i, intervalnumber), "\r",
            covevo_avg[i] = (covevo[i*startdist:i*startdist+intervallength] == covevo[i*startdist]).sum(axis=1)
    print ""
            
    result = covevo_avg.sum(axis=0)/float(covevo_avg.shape[0])
    
    for i in xrange(result.shape[0]):
        print result[i]

if __name__ == "__main__":
    main()
