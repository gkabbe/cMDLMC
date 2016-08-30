#!/usr/bin/env python3

import os
import sys
import time
import re

import ipdb
import numpy as np
import tables

from mdkmc.atoms import numpyatom as npa
from mdkmc.atoms.numpyatom import xyzatom as dtype_xyz
from mdkmc.misc.tools import chunk


class XYZFile(object):

    def __init__(self, filename, framenumber=None, verbose=False):
        self.filename = filename
        self.datei = open(self.filename, "rb")
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
            # line = self.datei.readline()
            # if line == "":
            # raise EOFError
            # self.datei.readline()
            # for index in range(self.atomnr):
            # line = self.datei.readline()
            # atoms[index]["name"] = line.split()[0]
            # atoms[index]["pos"][:] = list(map(float, line.split()[1:4]))
            lines = self.datei.readlines()
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
        start_time = time.time()
        traj = []
        counter = 0
        while 1:
            try:
                traj.append(self.get_atoms_numpy(atomnames))
            except EOFError:
                break
            if verbose and counter % 100 == 0:
                print("# Frame {}, ({:.2f} fps)".format(counter, float(
                    counter) / (time.time() - start_time)), "\r", end=' ')
            counter += 1
        if verbose:
            print("")
        if verbose:
            print("# Total time: {} sec".format(time.time() - start_time))

        return np.array(traj, dtype=npa.xyzatom)

    def get_frame_numpy(self, f, chunks=None):
        def filter_lines(f, frame_len):
            for i, line in enumerate(f):
                if i % frame_len not in (0, 1) and b"O" in line:
                    yield line

        data = np.genfromtxt(filter_lines(f, self.atomnr + 2), dtype=dtype_xyz,
                             max_rows=chunks * 144).reshape((chunks, -1))

        return data

    def get_trajectory_memmap_old(self, memmap_fname, atomnames=None, verbose=False):
        frame_number = self.get_framenumber()
        memmap = np.lib.format.open_memmap(memmap_fname, dtype=dtype_xyz,
                                           shape=(frame_number, self.atomnr), mode="w+")

        start_time = time.time()
        frames = []
        for i in range(frame_number):
            # memmap[i, :] = self.get_atoms_numpy(atomnames)
            frames.append(self.get_atoms_numpy(atomnames))
            if verbose and i > 0 and i % 1000 == 0:
                print("# {:06d} / {:06d} "
                      "... {:.1f} min remaining".format(i, frame_number,
                                                        (frame_number - i) / i *
                                                        (time.time() - start_time)
                                                        ),
                      end="\r")
                sys.stdout.flush()
        if verbose:
            print("")
        memmap[:] = frames
        return memmap

    def get_trajectory_memmap(self, memmap_fname, atomnames=None, verbose=False):
        if verbose:
            print("# Determining number of frames of trajectory")
        frame_number = self.get_framenumber()
        memmap = np.lib.format.open_memmap(memmap_fname, dtype=dtype_xyz,
                                           shape=(frame_number, 144), mode="w+")

        start_time = time.time()
        frames = []
        self.datei.seek(0)
        for start, stop in chunk(range(frame_number), 5000):
            memmap[start:stop] = self.get_frame_numpy(self.datei, stop - start)
            if verbose:
                print("# {:06d} / {:06d} "
                      "... {:.1f} min remaining".format(stop, frame_number,
                                                        (frame_number - stop) / stop *
                                                        (time.time() - start_time)
                                                        ),
                      end="\r")
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


def get_positions(xyz_fname):
    position_regex = r"""
    ^\s*\w+             # atom name before positions
    (\s*(?:(?:\-?\d+)?  # digit before decimal point
    \.                  # decimal point
    \d*\s*){3})$        # digits after decimal point"""

    with open(xyz_fname, "r") as f:
        atomnumber = int(f.readline())
        f.seek(0)
        first_frame = [f.readline() for i in range(atomnumber + 2)]
        # offsets.append(len("".join(first_frame[:2])))
        coordinates = "".join(first_frame[2:12])
        print(coordinates)
        offsets = re.findall(position_regex, coordinates, re.MULTILINE | re.VERBOSE)
    return "\n".join(offsets)
    # for match in offsets:
    # print(coordinates[match.start(1):match.end(1)], match.start(1), match.end(1))


# def parse(xyz_fname):
#    regex = re.compile(r"""
#    ^\s*(\w+)\s+
#    (\-?\d*\.\d*   # 1st coordinate
#    \s+
#    \-?\d*\.\d*    # 2nd coordinate
#    \s+
#    \-?\d*\.\d*)$  # 3rd coordinate
#    """, re.VERBOSE | re.MULTILINE)
#
#    with open(xyz_fname, "r") as f:
#        print(re.findall(regex, f.read()))
#        f.seek(0)
#        atomnumber = int(f.readline())
#        f.seek(0)
#        arr = np.fromregex(f, regex, dtype=dtype_xyz)
#
#    return arr.reshape((-1, atomnumber))


def parse(f, atom_number, chunk=None, filter_=None):
    def filter_lines(f, frame_len):
        for i, line in enumerate(f):
            if i % frame_len not in (0, 1):
                yield line
                
    if filter_:
        filter_ = filter_(filter_lines(f, atom_number + 2))
    else:
        filter_ = filter_lines(f, atom_number + 2)

    data = np.genfromtxt(filter_, dtype=dtype_xyz, 
                         max_rows=chunk * atom_number).reshape((-1, atom_number))

    return data

def save_trajectory_to_hdf5(xyz_fname, *atom_names):
    with open(xyz_fname, "rb") as f:
        atom_number = int(f.readline())
        f.seek(0)
        first_frame = parse(f, atom_number, chunk = 1)
        atom_nr_dict = dict()
        for name in atom_names:
            atom_nr_dict[name] = (first_frame[0]["name"] == name).sum()
    
    a = tables.Atom.from_dtype(np.dtype("float32"))
    filters = tables.Filters(complevel=5, complib="blosc")
    hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    with tables.open_file(hdf5_fname, "a") as f:
        f.create_group("/", filters=filters)
        for name in atom_names:
            f.create_earray("/", atom_name, atom=a, shape=(0, atom_nr_dict[name], 3))
