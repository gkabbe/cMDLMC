#!/usr/bin/python -u

import numpy as np
import ipdb
import os
import re
import argparse
import time

from mdkmc.IO import BinDump
from mdkmc.cython_exts.atoms import numpyatom as npa
from mdkmc.cython_exts.helper import analysis_helper as ah

def get_P_neighbors(O_frame, P_frame, pbc):
    P_neighbors = np.zeros(O_frame.shape[0], int)
    for i in xrange(O_frame.shape[0]):
        P_index = npa.nextNeighbor(O_frame[i], P_frame, pbc=pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors



def compare(trajectory_path, pbc, recalc=False, anglecut=np.pi/2, distcut=3.0):
    print "trying to open {}".format(trajectory_path)
    trajectory = np.load(trajectory_path)
    trajectory_length = trajectory.shape[0]
    P_number = (trajectory[0]["name"] == "P").sum()
    O_number = (trajectory[0]["name"] == "O").sum()
    Os = np.array(trajectory[trajectory["name"] == "O"].reshape(trajectory_length, O_number)["pos"])
    Ps = np.array(trajectory[trajectory["name"] == "P"].reshape(trajectory_length, P_number)["pos"])
    P_neighbors = get_P_neighbors(Os[0], Ps[0], pbc)
    Hs = BinDump.npget_acidHs(trajectory, pbc)
    H_number = Hs.shape[0]
    jumpmat = np.zeros((O_number, O_number), int)
    anglemat = np.zeros((O_number, O_number), int)
    covevo_filename = re.sub("\_nobackup", "", os.path.splitext(trajectory_path)[0]) + "_covevo.npy"
    covevo = np.load(covevo_filename)
    formatstr = "{:0"+str(int(np.round(np.log10(trajectory.shape[0]))))+"d}"

    for i in xrange(1, trajectory_length):
        for j in xrange(H_number):
            if covevo[i-i, j] != covevo[i, j]:
                jumpmat[covevo[i-i, j], covevo[i, j]] += 1
        if i % 1000 == 0:
            print formatstr.format(i), "\r",
    print ""
    jumpmat_filename = os.path.splitext(trajectory_path)[0] + "_jumpmat"
    if not os.path.exists(jumpmat_filename) or recalc:
        print "saving to {}".format(jumpmat_filename)
        np.savetxt(jumpmat_filename, jumpmat, fmt="%d")
    else:
        print "{} already exists".format(jumpmat_filename)
        jumpmat = np.loadtxt(jumpmat_filename, dtype=int)


    anglemat_filename = os.path.splitext(trajectory_path)[0] + "_anglemat"
    if not os.path.exists(anglemat_filename) or recalc:
        print "checking angle criterion"
        ah.get_anglematrix(Os, Ps, P_neighbors, pbc, anglemat, angle_cutoff=anglecut, distance_cutoff=distcut)
        print "saving to {}".format(anglemat_filename)
        np.savetxt(anglemat_filename, anglemat, fmt="%d")
    else:
        print "{} already exists".format(anglemat_filename)
        anglemat = np.loadtxt(anglemat_filename, dtype=int)

    jumpmat_bool = jumpmat != 0
    anglemat_bool = anglemat != 0

    allowed_transitions = zip(*np.where(anglemat_bool))

    counter = 0
    for transition in allowed_transitions:
        if not jumpmat_bool[transition]:
            print transition
            counter += 1
    print counter, "allowed transitions were counted, which did not show any jumps in the MD"

    # entries_jump = zip(*np.where(jumpmat_bool))
    # print "no. of observed jumps in MD:", len(entries_jump)
    # counter = 0
    # for entry in entries_jump:
    #     if anglemat_bool[entry]:
    #         counter += 1
    # print "ratio allowed jumps/observed jumps"
    # print "{:.2%}".format(float(counter)/len(entries_jump))
    #
    # entries_angle = zip(*np.where(anglemat_bool))
    # print "no. of allowed jumps by angle criterion:", len(entries_angle)
    # counter = 0
    # for entry in entries_angle:
    #     if jumpmat_bool[entry]:
    #         counter += 1
    # print "ratio observed jumps/allowed jumps"
    # print "{:.2%}".format(float(counter)/len(entries_angle))
    # # print "matrices agree in {:.2%}".format((jumpmat_bool == anglemat_bool).sum()/float(jumpmat.shape[0]*jumpmat.shape[1]))
    #
    # jumpset = set(entries_jump)
    # angleset = set(entries_angle)
    #
    # intersect = jumpset.intersection(angleset)
    # union = jumpset.union(angleset)
    # print len(intersect)/float(len(union))


def main(*args):
    parser=argparse.ArgumentParser(description="Check if angle criterion only allows jumps between atoms, between which "
                                               "proton jumps can be observed in the MD",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file", help="config file")
    parser.add_argument("pbc", type=float, nargs=3, help="config file")
    parser.add_argument("--anglecut", "-a", default=np.pi/2, type=float, help="angle cutoff")
    parser.add_argument("--distcut", "-d", default=3.0, type=float, help="distance cutoff")
    parser.add_argument("--confighelp", "-c", action="store_true", help="config file help")
    parser.add_argument("--recalc", "-r", action="store_true", help="Calculate anew")
    args = parser.parse_args()

    pbc = np.array(args.pbc)
    compare(args.file, pbc, recalc=args.recalc, anglecut=args.anglecut, distcut=args.distcut)