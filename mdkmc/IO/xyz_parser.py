#!/usr/bin/env python3

import numpy as np
import os
import sys
import ipdb
import time

from mdkmc.atoms import atomclass as ac
from mdkmc.atoms import numpyatom as npa
from mdkmc.atoms.numpyatom import xyzatom as dtype_xyz

class XYZFile(object):
    def __init__(self, filename, framenumber=None, verbose=False):
        self.filename = filename
        self.datei = open(self.filename, "r")
        self.atomnr = int(self.datei.readline().split()[0])
        self.atomdict = self.create_atomtypedict(verbose=verbose)
        self.datei.seek(0)
        self.frame = 0
        self.framenumber = framenumber

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.datei.close()

    def create_atomtypedict(self, verbose=False):
        atomdict = dict()
        self.datei.seek(0)
        self.datei.readline()
        self.datei.readline()
        for index in range(self.atomnr):
            line = self.datei.readline()
            atomname = line.split()[0]
            if atomname in list(atomdict.keys()):
                atomdict[atomname].add(index)
            else:
                if verbose:
                    print("# Adding", atomname, "to atomdict")
                atomdict[atomname] = set([index])
        self.datei.seek(0)

        return atomdict

    def parse_frame_np(self, pos_array, atomname):
        self.datei.readline()
        self.datei.readline()
        i = 0
        for index in range(self.atomnr):
            line = self.datei.readline()
            if index in self.atomdict[atomname]:
                pos_array[i] = list(map(float, line.split()[1:4]))
                i += 1

    def get_atoms_numpy(self, atomnames=None):
        # pdb.set_trace()
        if atomnames is None:
            atoms = np.zeros(self.atomnr, dtype=npa.xyzatom)
            line = self.datei.readline()
            if line == "":
                raise EOFError
            self.datei.readline()
            for index in range(self.atomnr):
                line = self.datei.readline()
                atoms[index]["name"] = line.split()[0]
                atoms[index]["pos"][:] = list(map(float, line.split()[1:4]))
        else:
            atomnr_selection = sum([len(self.atomdict[atomname]) for atomname in atomnames])
            atoms = np.zeros(atomnr_selection, dtype=npa.xyzatom)
            line = self.datei.readline()
            if line == "":
                raise EOFError
            self.datei.readline()
            j = 0
            for index in range(self.atomnr):
                line = self.datei.readline()
                # name = line.split()[0]
                for atomname, atom_indices in self.atomdict.items():
                    if atomname in atomnames and index in atom_indices:
                        atoms[j]["name"] = atomname
                        atoms[j]["pos"][:] = list(map(float, line.split()[1:4]))
                        j += 1
        self.frame += 1

        return atoms

    def get_trajectory_numpy(self, atomnames=None, verbose=False):
        # pdb.set_trace()
        start_time = time.time()
        traj = []
        counter = 0
        while 1:
            try:
                traj.append(self.get_atoms_numpy(atomnames))
            except EOFError:
                break
            if verbose and counter % 100 == 0:
                print("# Frame {}, ({:.2f} fps)".format(counter, float(counter) / (time.time() - start_time)), "\r", end=' ')
            counter += 1
        if verbose:
            print("")
        if verbose:
            print("# Total time: {} sec".format(time.time()-start_time))

        return np.array(traj, dtype=npa.xyzatom)

    def get_trajectory_memmap(self, memmap_fname, atomnames=None, verbose=False):
        frame_number = self.get_framenumber()
        memmap = np.lib.format.open_memmap(memmap_fname, dtype=dtype_xyz, shape=(frame_number, self.atomnr), mode="w+")

        for i in range(frame_number):
            memmap[i, :] = self.get_atoms_numpy(atomnames)
            if verbose and i % 1000 == 0:
                print("# {:06d} / {:06d}\r".format(i, frame_number), end=' ')
                sys.stdout.flush()
        if verbose:
            print("")
        return memmap

    def print_frame(self):
        for i in range(self.atomnr + 2):
            print(self.datei.readline()[:-1])

    def print_selection(self, selection):
        atomnr = 0
        for a in list(self.atomdict.keys()):
            if a in selection:
                atomnr += len(self.atomdict[a])
        print(atomnr)
        print("")
        self.datei.readline()
        self.datei.readline()
        for index in range(self.atomnr):
            line = self.datei.readline()
            if line.split()[0] in selection:
                print(line[:-1])

    def seek_frame(self, n):
        self.datei.seek(0)
        if n == 0:
            pass
        else:
            i = 1
            while i % ((self.atomnr + 2) * n + 1) != 0:
                self.datei.readline()
                i += 1
        self.frame = n

    def framejump(self, n):
        for i in range((self.atomnr + 2) * n):
            self.datei.readline()

    def get_framenumber(self, verbose=False):
        if self.framenumber is not None:
            return self.framenumber
        else:
            linenumber = int(os.popen("wc -l " + self.filename).read().split()[0])
            framenumber = linenumber // (self.atomnr + 2)
            if verbose:
                print("#{} frames".format(framenumber))
            return framenumber

    def rewind(self):
        self.datei.seek(0)
