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
    for i in range(covevo.shape[0]):
        if i % 1000 == 0:
            print("At frame {}".format(i), "\r", end=' ')
        change = np.where(covevo[i] != covevo[i-1])[0]
        occupations = get_occupations(PO3_groups, covevo[i])
        for PO3_index, occupation_number in enumerate(occupations):
            if occupation_number == 3:
                occupation_time[PO3_index] += 1
            elif occupation_time[PO3_index] != 0:
                occupation_times.append(occupation_time[PO3_index])
                occupation_time[PO3_index] = 0
    print("")
    for ot in occupation_times:
        print(ot)


def count_occupation_times_absolute(PO3_groups, PO3_index, covevo):
    """If there is any defect in the whole trajectory, count the time until there is no defect"""
    occupation_times = []
    for i in range(covevo.shape[0]):
        change = np.where(covevo[i] != covevo[i-1])[0]
        if len(change) > 0:
            print("At frame {}:".format(i))
        occupations = get_occupations(PO3_groups, covevo[i])
        for PO3_index, occupation_number in enumerate(occupations):
            if occupation_number == 3:
                occupation_time[PO3_index] += 1
            elif occupation_time[PO3_index] != 0:
                occupation_times.append(occupation_time[PO3_index])
                occupation_time[PO3_index] = 0
    print(occupation_times)
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
    for i in range(covevo.shape[0]):
        # change = np.where(covevo[i] != covevo[i-1])[0]
        occupations = get_occupations(PO3_groups, covevo[i])
        if 3 in occupations:
            if dissociated:
                counter += 1
            else:
                print("now dissociated! (Frame {})".format(i))
                dissociated = True
                counter = 1
        else:
            if dissociated:
                print("At frame {} not dissociated any more!".format(i))
                dissociation_times.append(counter)
                counter = 0
                dissociated = False
        if i % 1000 == 0:
            print("{} / {}".format(i, covevo.shape[0]))
            print(dissociation_times)
    return dissociation_times

def proton_oxygen_distance_correlation(covevo, PO3_groups, Os, Hs, pbc):
    for i in range(1, covevo.shape[0]):
        change = np.where(covevo[i] != covevo[i-1])[0]
        if len(change) > 0:
            print("At frame {}:".format(i))
            for H_index in change:
                print("jump from {} to {}".format(covevo[i-1, H_index], covevo[i, H_index]))
                # first check, how long the jumping proton will stay close to its new neighbor oxygen
                # then check if the jumping proton bonds covalently to its new oxygen in this time
                new_neighbor = covevo[i, H_index]
                old_neighbor = covevo[i-1, H_index]
                j, count = i, 0
                OH_dists = []
                while j < covevo.shape[0] and covevo[j, H_index] == new_neighbor:
                    OH_dists.append(npa.length(Hs[j, H_index], Os[j, new_neighbor], pbc))
                    j += 1
                min_dist = min(OH_dists)
                count = j - i
                print("proton {} stays close to oxygen {} for {} frames".format(H_index, new_neighbor, count))
                print("closest distance: {}".format(min_dist))
                # if covalent bond exists in interval, check what the other protons at this phosphonic group are doing:
                if min_dist <= 1.1:
                    PO3_group_index = np.where(PO3_groups == new_neighbor)[0][0]
                    O_indices = [new_neighbor]
                    H_indices = [H_index]
                    for Oind in PO3_groups[PO3_group_index]:
                        if Oind != new_neighbor:
                            O_indices.append(Oind)
                            H_ind = npa.nextNeighbor(Os[i, Oind], Hs[i], pbc=pbc)[0]
                            H_indices.append(H_ind)
                    for k in range(1, 3):
                        if npa.length(Os[i, O_indices[k]], Hs[i, H_indices[k]], pbc) > 1.3:
                            print("the other two oxygens are not covalently bonded")
                            break
                    else:
                        for j in range(i, i+count):
                            for k in range(3):
                                print(npa.length(Os[j, O_indices[k]], Hs[j, H_indices[k]], pbc), end=' ')
                            print("")
                else:
                    print("proton {} never comes close enough".format(H_index))


def proton_phosphor_distance_correlation(covevo, PO3_groups, Os, Ps, pbc):
    for i in range(1, covevo.shape[0]):
        change = np.where(covevo[i] != covevo[i-1])[0]
        if len(change) > 0:
            print("At frame {}:".format(i))
            for H_index in change:
                print("jump from {} to {}".format(covevo[i-1, H_index], covevo[i, H_index]))
                # first check, how long the jumping proton will stay close to its new neighbor oxygen
                # then check if the jumping proton bonds covalently to its new oxygen in this time
                new_neighbor = covevo[i, H_index]
                old_neighbor = covevo[i-1, H_index]
                j, count = i, 0
                OH_dists = []
                while j < covevo.shape[0] and covevo[j, H_index] == new_neighbor:
                    OH_dists.append(npa.length(Hs[j, H_index], Os[j, new_neighbor], pbc))
                    j += 1
                min_dist = min(OH_dists)
                count = j - i
                print("proton {} stays close to oxygen {} for {} frames".format(H_index, new_neighbor, count))
                print("closest distance: {}".format(min_dist))
                # if covalent bond exists in interval, check what the other protons at this phosphonic group are doing:
                if min_dist <= 1.1:
                    PO3_group_index = np.where(PO3_groups == new_neighbor)[0][0]
                    O_indices = [new_neighbor]
                    H_indices = [H_index]
                    for Oind in PO3_groups[PO3_group_index]:
                        if Oind != new_neighbor:
                            O_indices.append(Oind)
                            H_ind = npa.nextNeighbor(Os[i, Oind], Hs[i], pbc=pbc)[0]
                            H_indices.append(H_ind)
                    for k in range(1, 3):
                        if npa.length(Os[i, O_indices[k]], Hs[i, H_indices[k]], pbc) > 1.3:
                            print("the other two oxygens are not covalently bonded")
                            break
                    else:
                        for j in range(i, i+count):
                            for k in range(3):
                                print(npa.length(Os[j, O_indices[k]], Hs[j, H_indices[k]], pbc), end=' ')
                            print("")
                else:
                    print("proton {} never comes close enough".format(H_index))


def dissociation_times(PO3_groups, covevo):

    # print PO3_groups

    dissociation_times = jump_probabilities(PO3_groups, covevo)
    for dt in dissociation_times:
        print(dt)

# count_occupation_times(PO3_groups, covevo)

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

    BinDump.mark_acidic_protons(trajectory, pbc, verbose=True)
    atoms = npa.select_atoms(trajectory, "O", "P", "AH")
    Os = atoms["O"]
    Ps = atoms["P"]
    Hs = atoms["AH"]
    print("# Hs:", Hs.shape)
    # O_indices = np.where(trajectory[0]["name"] == "O")[0]
    # P_indices = np.where(trajectory[0]["name"] == "O")[0]
    O_indices = list(range(Os.shape[1]))
    P_indices = list(range(Ps.shape[1]))
    first_frame  = trajectory[0]["pos"]

    # print P_indices
    # print O_indices
    # print Os.shape
    # print first_frame.shape

    PO3_groups = []

    for P_index in P_indices:
        O_neighbors = sorted(O_indices, key=lambda O_index: npa.length(Ps[0, P_index], Os[0, O_index], pbc))
        PO3_groups.append(O_neighbors[:3])

    PO3_groups = np.array(PO3_groups)

    # dissociation_times(PO3_groups, covevo)

    # proton_oxygen_distance_correlation(covevo, PO3_groups, Os, Hs, pbc)
    count_occupation_times(PO3_groups, covevo)


if __name__ == "__main__":
    main()
