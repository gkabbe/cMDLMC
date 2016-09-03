#!/usr/bin/env python3

import numpy as np
import math
import sys
import ipdb

xyzatom = np.dtype([("name", np.str_, 2), ("pos", np.float64, (3,))])

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
    selections = []
    frames = xyzatom_traj.shape[0]
    for atomname in atomnames:
        traj = xyzatom_traj[xyzatom_traj["name"] == atomname]["pos"]
        atomnumber = xyzatom_traj[0][xyzatom_traj[0]["name"] == atomname].size
        selections.append(np.array(traj.reshape((frames, atomnumber, 3)), order="C"))
    if len(atomnames) > 1:
        return selections
    else:
        return selections[0]


def map_indices(frame, atomname):
    """Returns indices of atomtype in original trajectory"""
    indices = []
    for i, atom in enumerate(frame):
        if atom["name"] == atomname:
            indices.append(i)
    return indices


def numpy_print(atoms, names=None, outfile=None):
    if names is not None:
        atoms = atoms[atoms["name"] == name]
        print(sum([len(atoms[atoms["name"] == name]) for name in names]), file=outfile)
    else:
        print(len(atoms), file=outfile)
    if outfile is None:
        outfile = sys.stdout
    print("", file=outfile)
    for x in atoms:
        print("{:4} {: 20} {: 20} {: 20}".format("H" if x["name"] == "AH" else x["name"], x["pos"][0],
                                                 x["pos"][1], x["pos"][2]), file=outfile)


def distance(single_atom, many_atoms, pbc):
    diff = many_atoms - single_atom
    while (diff > pbc / 2).any():
        diff = np.where(diff > pbc / 2, diff - pbc, diff)
    while (diff < -pbc / 2).any():
        diff = np.where(diff < -pbc / 2, diff + pbc, diff)
    return diff


def distance_pbc_nonortho(a1_pos, a2_pos, pbc):
    r = a2_pos - a1_pos
    h = pbc.reshape((3, 3)).T
    h_inv = np.linalg.inv(h)

    s = np.dot(h_inv, r)
    for i in range(3):
        s[i] -= round(s[i])
    r = np.dot(h, s)
    return r


def squared_distance(a1_pos, a2_pos, pbc, axis_wise=False):
    """Calculate squared distance using numpy vector operations"""
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


def sqdist_np_multibox(a1_pos, a2_pos, pbc, axis_wise=False):
    dist = a1_pos - a2_pos
    move_mat = np.zeros((dist.shape[0], dist.shape[1], dist.shape[2]))
    while (dist > pbc / 2).any():
        indices = np.where(dist > pbc / 2)
        dist[indices] -= pbc[indices[-1]]
        move_mat[indices] += 1
    while (dist < -pbc / 2).any():
        indices = np.where(dist < -pbc / 2)
        dist[indices] += pbc[indices[-1]]
        move_mat[indices] -= 1
    dist = move_mat * pbc + dist
    if axis_wise:
        return dist * dist
    else:
        return (dist * dist).sum(axis=1)


def length(a1_pos, a2_pos, pbc=None):
    dist = distance(a1_pos, a2_pos, pbc)
    return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])


def next_neighbor(a1_pos, atoms_pos, nonortho=False, pbc=None):
    mindist = 1e6
    if nonortho:
        for i in range(atoms_pos.shape[0]):
            diff = distance_pbc_nonortho(a1_pos, atoms_pos[i], pbc)
            dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
            if dist < mindist:
                mindist = dist
                minind = i
    else:
        diff = distance(a1_pos, atoms_pos, pbc)
        sq_dist = np.sum(diff * diff, axis=-1)
        min_index = np.argmin(sq_dist)
    return min_index, atoms_pos[min_index]


# angle between a1--a2 and a2--a3
def angle(a1_pos, a2_pos, a3_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a2_a3 = distance(a2_pos, a3_pos, pbc)
    return np.degrees(np.arccos(np.dot(a1_a2, a2_a3) / np.linalg.norm(a1_a2) / np.linalg.norm(a2_a3)))


def angle_vectorized(a1_pos, a2_pos, a3_pos, pbc):
    a1_a2 = distance(a1_pos, a2_pos, pbc)
    a2_a3 = distance(a2_pos, a3_pos, pbc)
    return np.degrees(np.arccos(
        np.einsum("ij, ij -> i", a1_a2, a2_a3) / np.sqrt(np.einsum("ij,ij->i", a1_a2, a1_a2)) / np.sqrt(
            np.einsum("ij,ij->i", a2_a3, a2_a3))))


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
        nn_index, next_neighbor = next_neighbor(H["pos"], not_H_atoms["pos"], nonortho=nonortho, pbc=pbc)
        # ~ipdb.set_trace()
        if not_H_atoms["name"][nn_index] == "O":
            acid_indices.append(H_indices[i])
            # ~ else:
            # ~ ipdb.set_trace()
    if verbose:
        print("# Acidic indices: ", acid_indices)
        print("# Number of acidic protons: ", len(acid_indices))
    return acid_indices


def remove_center_of_mass_movement(npa_traj):
    mass_array = np.array([atom_masses[name] for name in npa_traj[0]["name"]])[None, :, None]
    center_of_mass = (mass_array * npa_traj["pos"]).sum(axis=1)[:, None, :] / mass_array.sum()
    npa_traj["pos"] -= center_of_mass


def print_center_of_mass(npa_traj):
    mass_array = np.array([atom_masses[name] for name in npa_traj[0]["name"]])[None, :, None]
    center_of_mass = (mass_array * npa_traj["pos"]).sum(axis=1)[:, None, :] / mass_array.sum()
    for i, com in enumerate(center_of_mass):
        print("Frame {:6d}".format(i), com)


def print_center_of_mass_commandline(*args):
    trajectory = np.load(sys.argv[1])["trajectory"]
    print_center_of_mass(trajectory)
