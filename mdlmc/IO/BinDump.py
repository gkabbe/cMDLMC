# coding=utf-8

#!/usr/bin/env python3

import os
import re

import numpy as np
from mdlmc.atoms import numpy_atom as npa
from mdlmc.cython_exts.atoms import numpyatom as cython_npa


def npsave_atoms(np_filename, trajectory, nobackup=True, verbose=False):
    if compressed:
        suffix = ".npz"
    else:
        suffix = ".npy"

    np_filename = re.sub("\.npz$|\.npy$|\.xyz$", "", np_filename)

    if nobackup and "nobackup" not in np_filename:
        if verbose:
            print("# Adding _nobackup to filename")
        np_filename += "_nobackup"

    if not overwrite:
        while os.path.exists(np_filename + suffix):
            if verbose:
                print("# Filename {} already existing".format(np_filename + suffix))
            number = re.findall("(\((\d\d)\)){,1}$", np_filename)[0][1]
            if number == "":
                np_filename += "(01)"
            else:
                number = int(number) + 1
                np_filename = re.sub("\(\d\d\)", "({:02d})".format(number), np_filename)

    np_filename += suffix

    if verbose:
        print("# Saving to {}".format(np_filename))

    if compressed:
        np.savez_compressed(np_filename, trajectory)
    else:
        np.save(np_filename, trajectory)


def npload_atoms(filename, arg="arr_0", atomnames_list=None, remove_com=True,
                 create_if_not_existing=False, return_tuple=False, verbose=True):
    def find_binfiles(filename):
        binfiles = []
        for ending in [".npy", ".npz"]:
            for middle in ["", "_nobackup"]:
                if os.path.exists(filename[:-4] + middle + ending):
                    binfiles.append(filename[:-4] + middle + ending)
        return binfiles

    if os.path.splitext(filename)[1] == ".xyz":
        binfiles = find_binfiles(filename)
        if len(binfiles) > 0:
            if verbose:
                print("# Found following binary files: {}".format(binfiles))
            trajectory = np.load(binfiles[0])
            if binfiles[0][-4:] == ".npz" and arg in list(trajectory.keys()):
                trajectory = trajectory[arg]
            if atomnames_list is not None:
                selection = None
                atomnumber = 0
                for atomname in atomnames_list:
                    if selection is None:
                        selection = trajectory["name"] == atomname
                    else:
                        selection = np.logical_or(selection, trajectory["name"] == atomname)
                    atomnumber += (trajectory[0]["name"] == atomname).sum()
                trajectory = trajectory[selection]
                trajectory = trajectory.reshape(trajectory.shape[0] / atomnumber, atomnumber)
        else:
            if verbose:
                print("# Found no binary files!")
                print("# Loading from file {}".format(filename))
            xyz_file = XYZFile(filename, verbose=verbose)
            trajectory = xyz_file.get_trajectory_numpy(atomnames_list, verbose=verbose)
            if remove_com:
                npa.remove_com_movement_traj(trajectory, verbose=verbose)
            if create_if_not_existing:
                npsave_atoms(filename, trajectory, verbose=verbose)
    else:
        if filename[-4:] == ".npy":
            trajectory = np.load(filename)
        elif filename[-4:] == ".npz":
            if verbose:
                print("# Compressed binary file archive. Loading {}.".format(arg))
            trajectory = np.load(filename)[arg]

    if return_tuple and atomnames_list is not None:
        atomlist = []
        for atomname in atomnames_list:
            atomnr = (trajectory[0]["name"] == atomname).sum()
            traj = trajectory[trajectory["name"] == atomname].reshape((trajectory.shape[0], atomnr))
            atomlist.append(traj)
        return atomlist
    else:
        return trajectory


def npload_memmap(filename, verbose=False):
    root, ext = os.path.splitext(filename)
    if ext == ".xyz":
        if "nobackup" not in root:
            memmap_filename = "_".join([root, "nobackup.npy"])
        else:
            memmap_filename = "".join([root, ".npy"])
    else:
        memmap_filename = filename

    try:
        memmap = np.lib.format.open_memmap(memmap_filename, mode="r")
    except (IOError, FileNotFoundError):
        if verbose:
            print("# Loading file {}".format(filename))
        xyz_file = XYZFile(filename, verbose=verbose)
        if verbose:
            print("# Saving new mem map under {}".format(memmap_filename))
        memmap = xyz_file.get_trajectory_memmap(memmap_filename, verbose=verbose)
        npa.remove_com_movement_traj(memmap, verbose=verbose)
    return memmap, memmap_filename


def mark_acidic_protons(traj, pbc, nonortho=False, verbose=False):
    indices = cython_npa.get_acidic_proton_indices(traj[0], pbc, nonortho=nonortho, verbose=verbose)
    traj["name"][:, indices] = "AH"
