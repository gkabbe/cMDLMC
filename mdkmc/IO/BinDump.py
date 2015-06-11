#!/usr/bin/python

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
from mdkmc.IO.xyzparser import XYZFile
from mdkmc.atoms import atomclass
from mdkmc.atoms import numpyatom as npa
from mdkmc.misc.timer import TimeIt


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
        print "Expecting array with dimensions (No of frames, no of atoms) or (no of atoms)"
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
        return Hs[acidH_indices]


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
            print "# Adding _nobackup to filename"
        np_filename += "_nobackup"

    if not overwrite:
        while os.path.exists(np_filename+suffix):
            if verbose:
                print "# sFilename {} already existing".format(np_filename+suffix)
            number = re.findall("(\((\d\d)\)){,1}$", np_filename)[0][1]
            if number == "":
                np_filename += "(01)"
            else:
                number = int(number) + 1
                np_filename = re.sub("\(\d\d\)", "({:02d})".format(number), np_filename)

    np_filename += suffix

    if verbose:
        print "# Saving to {}".format(np_filename)

    if compressed:
        np.savez_compressed(np_filename, trajectory)
    else:
        np.save(np_filename, trajectory)


def npload_atoms(filename, arg="arr_0", atomnames_list=[], remove_com=True, create_if_not_existing=False, verbose=False):
    def find_binfiles(filename):
        binfiles = []
        for ending in [".npy", ".npz"]:
            for middle in ["", "_nobackup"]:
                if os.path.exists(filename[:-4] + middle + ending):
                    binfiles.append(filename[:-4] + middle + ending)
        return binfiles

    if filename[-4:] == ".xyz":
        binfiles = find_binfiles(filename)
        if verbose:
            print "# Found following binfiles: {}".format(binfiles)
        if len(binfiles) > 0:
            traj = np.load(binfiles[0])
            if binfiles[0][-4:] == ".npz" and arg in traj.keys():
                traj = traj[arg]
            if len(atomnames_list) > 0:
                selection = None
                atomnumber = 0
                for atomname in atomnames_list:
                    if selection is None:
                        selection = traj["name"] == atomname
                    else:
                        selection = np.logical_or(selection, traj["name"] == atomname)
                    atomnumber += (traj[0]["name"] == atomname).sum()
                traj = traj[selection]
                traj = traj.reshape(traj.shape[0]/atomnumber, atomnumber)
        else:
            if verbose:
                print "# Loading from file {}".format(filename)
            xyz_file = XYZFile(filename, verbose=verbose)
            traj = xyz_file.get_trajectory_numpy(atomnames_list, verbose=verbose)
            if remove_com:
                npa.remove_com_movement_traj(traj, verbose=verbose)
            if create_if_not_existing:
                npsave_atoms(filename, traj, verbose=verbose)
    else:
        if filename[-4:] == ".npy":
            traj = np.load(filename)
        elif filename[-4:] == ".npz":
            if verbose:
                print "# Compressed binary file archive. Loading {}.".format(arg)
            traj = np.load(filename)[arg]

    return traj


def mark_acidic_protons(traj, pbc, nonortho=False, verbose=False):
    indices = cython_npa.get_acidic_proton_indices(traj[0], pbc, nonortho=nonortho, verbose=verbose)
    traj["name"][:, indices] = "AH"


def npsave_covevo(fname, Os, Hs, pbc, nonortho=False, verbose=False):
    print "# Determining covevo.."
    start_time = time.time()
    covevo = np.zeros((Hs.shape[0], Hs.shape[1]), int)

    if nonortho:
        if verbose:
            print "# Nonorthogonal periodic boundaries"
        hmat = np.array(pbc.reshape((3, 3)).T, order="C")
        hmat_inv = np.array(np.linalg.inv(hmat), order="C")

        for i in xrange(covevo.shape[0]):
            for j in xrange(covevo.shape[1]):
                covevo[i, j] = cython_npa.nextNeighbor_nonortho(Hs[i, j], Os[i], hmat, hmat_inv)[0]
            if verbose and i % 100 == 0:
                print "# Frame {: 6d} ({:5.0f} fps)".format(i, float(i)/(time.time()-start_time)),
                print "\r",
        print ""
    else:
        for i in xrange(covevo.shape[0]):
            for j in xrange(covevo.shape[1]):
                covevo[i, j] = cython_npa.nextNeighbor(Hs[i, j], Os[i], pbc=pbc)[0]
            if verbose and i % 100 == 0:
                print "# Frame {: 6d} ({:5.0f} fps)".format(i, float(i)/(time.time()-start_time)),
                print "\r",
        print ""

    if verbose:
        print "# Saving covevo in {}".format(fname)
    np.save(fname, covevo)
    if verbose:
        print "# Total time: {} seconds".format(time.time()-start_time)
    return covevo
