#!/usr/bin/env python3

import sys
import os
import numpy as np
import pickle
import ipdb
import time
import re
import inspect

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#~ sys.path.append(os.path.join(script_path, "../cython/atoms"))
from mdkmc.cython_exts.atoms import numpyatom as cython_npa
from mdkmc.IO.xyz_parser import XYZFile
from mdkmc.atoms import numpyatom as npa


def get_acidHs(atoms, pbc):
    Hs = []
    acidHs = []
    for atom in atoms:
        if atom.name == "H":
            Hs.append(atom)

    for H in Hs:
        if (H.next_neighbor(atoms, pbc))[0].name == "O":
            acidHs.append(H.index)

    return acidHs


def npget_acidHs(atoms, pbc, only_indices=False):
    if len(atoms.shape) > 2 or len(atoms.shape) == 0:
        print("Expecting array with dimensions (No. of frames, no. of atoms) or (no. of atoms)")
    elif len(atoms.shape) == 2:
        atoms = atoms[0]
    else:
        pass
    acidH_indices = []
    atoms_pos = np.array(atoms["pos"], order="C")
    for i, atom in enumerate(atoms_pos):
        if atoms[i]["name"] == "H" and atoms[cython_npa.nextNeighbor(atom, atoms_pos, pbc, exclude_identical_position=True)[0]]["name"] == "O":
            acidH_indices.append(i)
    if only_indices:
        return acidH_indices
    else:
        return atoms_pos[acidH_indices]


def get_Os(atoms):
    Os=[]
    for atom in atoms:
        if atom.name == "O":
            Os.append(atom.index)
    return Os


def npsave_atoms(np_filename, trajectory, overwrite=False, compressed=False, nobackup=True, verbose=False):

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
        while os.path.exists(np_filename+suffix):
            if verbose:
                print("# sFilename {} already existing".format(np_filename+suffix))
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
                trajectory = trajectory.reshape(trajectory.shape[0]/atomnumber, atomnumber)
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
        if not "nobackup" in root:
            memmap_filename = "_".join([root, "nobackup.npy"])
        else:
            memmap_filename = "".join([root, ".npy"])
    else:
        memmap_filename = filename

    try:
        memmap = np.lib.format.open_memmap(memmap_filename, mode="r")
    except IOError:
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


def npsave_covevo(fname, Os, Hs, pbc, nonortho=False, verbose=False):
    """Saves the evolution of the covalent bonds of the hydrogen atoms with the oxygen atoms -> covevo"""
    print("# Determining covevo..")
    start_time = time.time()
    covevo = np.zeros((Hs.shape[0], Hs.shape[1]), int)

    if nonortho:
        if verbose:
            print("# Nonorthogonal periodic boundaries")
        hmat = np.array(pbc.reshape((3, 3)).T, order="C")
        hmat_inv = np.array(np.linalg.inv(hmat), order="C")

        for i in range(covevo.shape[0]):
            for j in range(covevo.shape[1]):
                covevo[i, j] = cython_npa.nextNeighbor_nonortho(Hs[i, j], Os[i], hmat, hmat_inv)[0]
            if verbose and i % 100 == 0:
                print("# Frame {: 6d} ({:5.0f} fps)".format(i, float(i)/(time.time()-start_time)), end=' ')
                print("\r", end=' ')
        print("")
    else:
        for i in range(covevo.shape[0]):
            for j in range(covevo.shape[1]):
                covevo[i, j] = cython_npa.nextNeighbor(Hs[i, j], Os[i], pbc=pbc)[0]
            if verbose and i % 100 == 0:
                print("# Frame {: 6d} ({:5.0f} fps)".format(i, float(i)/(time.time()-start_time)), end='\r')
                print("")
        print("")

    if verbose:
        print("# Saving covevo in {}".format(fname))
    np.save(fname, covevo)
    if verbose:
        print("# Total time: {} seconds".format(time.time()-start_time))
    return covevo
