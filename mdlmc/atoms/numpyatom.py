#!/usr/bin/env python3

import sys
import numpy as np

from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic

dtype_xyz = np.dtype([("name", np.str_, 2), ("pos", np.float64, (3,))])

atom_masses = {'C': 12.001,
               'Cl': 35.45,
               'Cs': 132.90545196,
               'H': 1.008,
               'O': 15.999,
               'P': 30.973761998,
               'S': 32.06,
               'Se': 78.971}


def get_acidic_protons(frame, atom_box, verbose=False):
    """Returns the indices of all protons whose closest neighbor is an oxygen atom"""
    if len(frame.shape) > 1:
        raise ValueError("Argument frame should be a one dimensional numpy array of type dtype_xyz")
    acidic_indices = []
    H_atoms = np.array(frame["pos"][frame["name"] == "H"])
    H_indices = np.where(frame["name"] == "H")[0]
    not_H_atoms = np.array(frame["pos"][frame["name"] != "H"])
    not_H_atoms_names = frame["name"][frame["name"] != "H"]
    for i, H in enumerate(H_atoms):
        nn_index, next_neighbor = atom_box.next_neighbor(H, not_H_atoms)
        if not_H_atoms_names[nn_index] == "O":
            acidic_indices.append(H_indices[i])
    if verbose:
        print("# Acidic indices: ", acidic_indices)
        print("# Number of acidic protons: ", len(acidic_indices))
    return acidic_indices


def select_atoms(xyzatom_traj, *atomnames):
    """Select atoms from a trajectory of dtype \"dtype_xyz\"."""
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
        print(
            "{:4} {: 20} {: 20} {: 20}".format("H" if x["name"] == "AH" else x["name"], x["pos"][0],
                                               x["pos"][1], x["pos"][2]), file=outfile)


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
