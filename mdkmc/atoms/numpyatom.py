#!/usr/bin/python

import numpy as np
import math
import sys
import ipdb


xyzatom = np.dtype([("name", np.str_, 2), ("pos", np.float64, (3,))])
pdbatom = np.dtype(
    [("rec", np.str_, 4), ("serialnr", np.int32), ("name", np.str_, 4), ("locind", np.str_, 1), ("resname", np.str_, 3),
     ("chainid", np.str_, 1), ("rsn", np.int32), \
     ("rescode", np.str_, 1), ("pos", np.float64, (3,)), ("occ", np.float64), ("tempfac", np.float64),
     ("segment", np.str_, 4), ("element", np.str_, 2), ("charge", np.str_, 2)])

atom_masses = dict()
atom_masses["H"] = 1.008
atom_masses["C"] = 12.001
atom_masses["O"] = 15.999
atom_masses["P"] = 30.973761998
atom_masses["S"] = 32.06
atom_masses["Cl"] = 35.45
atom_masses["Cs"] = 132.90545196
atom_masses["Se"] = 78.971


def select_atoms(xyzatom_traj, *atomnames):
    selections = dict()
    frames = xyzatom_traj.shape[0]
    for atomname in atomnames:
        traj = xyzatom_traj[xyzatom_traj["name"] == atomname]["pos"]
        atomnumber = xyzatom_traj[0][xyzatom_traj[0]["name"] == atomname].size
        selections[atomname] = np.array(traj.reshape((frames, atomnumber, 3)), order="C")
    return selections


def numpyprint(atoms, names=None, outfile=None):
    if names != None:
        atoms = atoms[atoms["name"] == name]
        print >> outfile, sum([len(atoms[atoms["name"] == name]) for name in names])
    else:
        print >> outfile, len(atoms)
    if outfile == None:
        outfile = sys.stdout
    print >> outfile, ""
    for x in atoms:
        print >> outfile, "{:4} {: 20} {: 20} {: 20}".format("H" if x["name"] == "AH" else x["name"], x["pos"][0],
                                                             x["pos"][1], x["pos"][2])


def distance(a1_pos, a2_pos, pbc=None):
    dist = a2_pos - a1_pos
    if pbc is not None:
        for i in xrange(3):
            while dist[i] < -pbc[i] / 2:
                dist[i] += pbc[i]
            while dist[i] > pbc[i] / 2:
                dist[i] -= pbc[i]
    return dist


def distance_pbc_nonortho(a1_pos, a2_pos, pbc):
    r = a2_pos - a1_pos
    h = pbc.reshape((3, 3)).T
    h_inv = np.linalg.inv(h)

    s = np.dot(h_inv, r)
    for i in range(3):
        s[i] -= round(s[i])
    r = np.dot(h, s)
    return r


def distance_pbc_nonortho_sort(a1_pos, a2_pos, pbc):
    pass


def distance_pbc_nonortho_new(a1_pos, a2_pos, pbc):
    h = pbc.reshape((3, 3)).T
    h_inv = np.linalg.inv(h)

    lengths = [(i, np.sqrt(np.dot(v, v))) for i, v in enumerate(pbc.reshape((3, 3)))]
    lengths.sort(key=lambda x: -x[1])

    a1_s = np.dot(h_inv, a1_pos)
    a2_s = np.dot(h_inv, a2_pos)

    sdiff = a2_s - a1_s
    for i in range(3):
        sdiff[i] = sdiff[i] - round(sdiff[i])

    rdiff = np.dot(h, sdiff)
    return rdiff


def distance_pbc_nonortho_bruteforce(a1_pos, a2_pos, pbc):
    diffvec = distance_pbc_nonortho(a1_pos, a2_pos, pbc)
    pbc = pbc.reshape((3, 3))
    dists = []
    for i in xrange(-1, 2):
        for j in xrange(-1, 2):
            for k in xrange(-1, 2):
                d = diffvec + i * pbc[0] + j * pbc[1] + k * pbc[2]
                dists.append((np.sqrt(np.dot(d, d)), d))
                print dists[-1][0]
    return min(dists, key=lambda x: x[0])[1]


# Calculate squared distance
def sqdist(a1_pos, a2_pos, pbc=None):
    dist = a1_pos - a2_pos
    for i in xrange(3):
        while dist[i] < -pbc[i] / 2:
            dist[i] += pbc[i]
        while dist[i] > pbc[i] / 2:
            dist[i] -= pbc[i]

    return dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]


# Calculate squared distance using numpy vector operations
def sqdist_np(a1_pos, a2_pos, pbc, axis_wise=False):
    dist = a1_pos - a2_pos
    while (dist > pbc / 2).any():
        indices = np.where(dist > pbc / 2)
        dist[indices] -= pbc[indices[-1]]
    while (dist < -pbc / 2).any():
        indices = np.where(dist < -pbc / 2)
        dist[indices] += pbc[indices[-1]]
    if axis_wise:
        return dist * dist
    else:
        return (dist * dist).sum(axis=1)


def length(a1_pos, a2_pos, pbc=None):
    dist = distance(a1_pos, a2_pos, pbc)
    return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])


def nextNeighbor(a1_pos, atoms_pos, nonortho=False, pbc=None):
    mindist = 1e6
    for i in xrange(atoms_pos.shape[0]):
        if nonortho == True:
            diff = distance_pbc_nonortho(a1_pos, atoms_pos[i], pbc)
        else:
            diff = distance(a1_pos, atoms_pos[i], pbc)
        dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
        if dist < mindist:
            mindist = dist
            minind = i
    return minind, atoms_pos[i]


# angle between a1--a2 and a2--a3
def angle(a1_pos, a2_pos, a3_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a2_a3 = distance(a2_pos, a3_pos, pbc)
    return np.degrees(np.arccos(np.dot(a1_a2, a2_a3) / np.linalg.norm(a1_a2) / np.linalg.norm(a2_a3)))


# angle between a1--a2 and a3--a4
def angle_4(a1_pos, a2_pos, a3_pos, a4_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a3_a4 = distance(a3_pos, a4_pos, pbc)
    return np.degrees(np.arccos(np.dot(a1_a2, a3_a4) / np.linalg.norm(a1_a2) / np.linalg.norm(a3_a4)))


# angle between a1--a2 and a2--a3
def angle_rad(a1_pos, a2_pos, a3_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a2_a3 = distance(a2_pos, a3_pos, pbc)
    return np.arccos(np.dot(a1_a2, a2_a3) / np.linalg.norm(a1_a2) / np.linalg.norm(a2_a3))


# angle between a1--a2 and a3--a4
def angle_4_rad(a1_pos, a2_pos, a3_pos, a4_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a3_a4 = distance(a3_pos, a4_pos, pbc)
    return np.arccos(np.dot(a1_a2, a3_a4) / np.linalg.norm(a1_a2) / np.linalg.norm(a3_a4))


def get_acidic_proton_indices(atoms, pbc=None, nonortho=False, verbose=False):
    acid_indices = []
    H_atoms = atoms[atoms["name"] == "H"]
    H_indices = np.where(atoms["name"] == "H")[0]
    not_H_atoms = atoms[atoms["name"] != "H"]
    for i, H in enumerate(H_atoms):
        nn_index, next_neighbor = nextNeighbor(H["pos"], not_H_atoms["pos"], nonortho=nonortho, pbc=pbc)
        # ~ipdb.set_trace()
        if not_H_atoms["name"][nn_index] == "O":
            acid_indices.append(H_indices[i])
            # ~ else:
            # ~ ipdb.set_trace()
    if verbose == True:
        print "#acidic indices: ", acid_indices
        print "#number of acidic protons: ", len(acid_indices)
    return acid_indices


def remove_com_movement_frame(npa_frame, verbose=False):
    com = np.zeros(3)
    M = 0
    for atom in npa_frame:
        com += atom_masses[atom["name"]] * atom["pos"]
        M += atom_masses[atom["name"]]
    com /= M
    npa_frame["pos"] -= com


def remove_com_movement_traj(npa_traj, verbose=False):
    if verbose == True:
        print "#Removing Center of Mass Movement"
    for i, frame in enumerate(npa_traj):
        if verbose == True and i % 1000 == 0:
            print "#Frame {} / {}".format(i, npa_traj.shape[0]), "\r",
        remove_com_movement_frame(frame, verbose=verbose)
    if verbose == True:
        print ""
