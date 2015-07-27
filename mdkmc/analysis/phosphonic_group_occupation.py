import numpy as np
import argparse
import glob
import re
import sys

from mdkmc.IO import BinDump
from mdkmc.atoms import numpyatom as npa

def get_occupations(PO3_groups, covevo):
    occupations = np.zeros(PO3_groups.shape[0])
    for H_index, O_index in enumerate(covevo):
        PO3_index = np.where(PO3_groups == O_index)[0][0]
        occupations[PO3_index] += 1
    return occupations

def count_occupation_times(PO3_groups, covevo):
    # occupations = np.zeros(PO3_groups.shape[0])
    occupation_time = np.zeros(PO3_groups.shape[0])
    occupation_times = []
    for i in xrange(covevo.shape[0]):
        change = np.where(covevo[i] != covevo[i-1])[0]
        if len(change) > 0:
            print "At frame {}:".format(i)
        occupations = get_occupations(PO3_groups, covevo[i])
        for PO3_index, occupation_number in enumerate(occupations):
            if occupation_number == 3:
                occupation_time[PO3_index] += 1
            elif occupation_time[PO3_index] != 0:
                occupation_times.append(occupation_time[PO3_index])
                occupation_time[PO3_index] = 0
    print occupation_times

def count_occupation_times_absolute(PO3_groups, PO3_index, covevo):
    """If there is any defect in the whole trajectory, count the time until there is no defect"""
    occupation_times = []
    for i in xrange(covevo.shape[0]):
        change = np.where(covevo[i] != covevo[i-1])[0]
        if len(change) > 0:
            print "At frame {}:".format(i)
        occupations = get_occupations(PO3_groups, covevo[i])
        for PO3_index, occupation_number in enumerate(occupations):
            if occupation_number == 3:
                occupation_time[PO3_index] += 1
            elif occupation_time[PO3_index] != 0:
                occupation_times.append(occupation_time[PO3_index])
                occupation_time[PO3_index] = 0
    print occupation_times
            # for c in change:
                # start_PO3_group = np.where(PO3_groups == covevo[i-1, c])[0][0]
                # destination_PO3_group = np.where(PO3_groups == covevo[i, c])[0][0]
                # print "H {} jumped from O {} to O {}".format(c, covevo[i-1, c], covevo[i, c])
                # print "H {} jumped from group {} to group {}".format(c, start_PO3_group, destination_PO3_group)
    # print occupations/covevo.shape[0]

def jump_probabilities(PO3_groups, covevo):
    """Determine the jump probabilities when a proton jumps from a P(OH)3, and when it jumps from
    a PO(OH)2"""
    dissociated = False
    dissociation_times = []
    for i in xrange(covevo.shape[0]):
        # change = np.where(covevo[i] != covevo[i-1])[0]
        occupations = get_occupations(PO3_groups, covevo[i])
        if 3 in occupations:
            if dissociated:
                counter += 1
            else:
                print "now dissociated! (Frame {})".format(i)
                dissociated = True
                counter = 1
        else:
            if dissociated:
                print "At frame {} not dissociated any more!".format(i)
                dissociation_times.append(counter)
                counter = 0
                dissociated = False
        if i % 1000 == 0:
            print "{} / {}".format(i, covevo.shape[0])
            print dissociation_times
    return dissociation_times



def main(*args):
    parser=argparse.ArgumentParser(description="Phosphonic Group Occupation Counter", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trajectory", help="Trajectory")
    parser.add_argument("pbc", type=float, nargs=3, help="Trajectory")
    args = parser.parse_args()

    trajectory = BinDump.npload_atoms(args.trajectory)
    covevo_filename = glob.glob(re.sub("_nobackup.npy", "", args.trajectory)+"*"+"covevo.npy")[0]
    # print covevo_filename
    covevo = np.load(covevo_filename)
    pbc = np.array(args.pbc)

    Ps = trajectory[0][trajectory[0]["name"] == "P"]["pos"]
    Os = trajectory[0][trajectory[0]["name"] == "O"]["pos"]
    # O_indices = np.where(trajectory[0]["name"] == "O")[0]
    # P_indices = np.where(trajectory[0]["name"] == "O")[0]
    O_indices = range(Os.shape[0])
    P_indices = range(Ps.shape[0])
    first_frame  = trajectory[0]["pos"]

    # print P_indices
    # print O_indices
    # print Os.shape
    # print first_frame.shape

    PO3_groups = []

    for P_index in P_indices:
        O_neighbors = sorted(O_indices, key=lambda O_index: npa.length(Ps[P_index], Os[O_index], pbc))
        PO3_groups.append(O_neighbors[:3])

    PO3_groups = np.array(PO3_groups)
    # print PO3_groups

    dissociation_times = jump_probabilities(PO3_groups, covevo)
    for dt in dissociation_times:
        print dt

# count_occupation_times(PO3_groups, covevo)

if __name__ == "__main__":
    main()
