#!/usr/bin/python

import numpy as np
import os
import ipdb
import time

from mdkmc.atoms import atomclass as ac
from mdkmc.atoms import numpyatom as npa
from mdkmc.misc.timer import TimeIt

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
        for index in xrange(self.atomnr):
            line = self.datei.readline()
            atomname = line.split()[0]
            if atomname in atomdict.keys():
                atomdict[atomname].add(index)
            else:
                if verbose:
                    print "#adding", atomname, "to atomdict"
                atomdict[atomname] = set([index])
        self.datei.seek(0)

        return atomdict

    def parse_frame_Os(self):
        Os=[]
        self.datei.readline()
        self.datei.readline()
        for index in xrange(self.atomnr):
            line = self.datei.readline()
            if index in self.atomdict["O"]:
                Os.append(ac.Atom(line.split()[0],map(float,line.split()[1:4]),index))
        return Os

    def parse_frame_Os_np(self, pos_array):
        self.datei.readline()
        self.datei.readline()
        i=0
        for index in xrange(self.atomnr):
            line = self.datei.readline()
            if index in self.atomdict["O"]:
                pos_array[i] = map(float, line.split()[1:4])
                i += 1

    def parse_frames_np_inplace(self, arr, atomname, verbose=False):
        """Expects numpy array in the shape (frames, oxygennumber, 3)"""
        frames = arr.shape[0]
        for frame in xrange(frames):
            line = self.datei.readline()
            if line == "":
                return frame
            if frame % 1000 == 0:
                if verbose:
                    print "#Frame", frame,
                    print "\r",
            self.datei.readline()
            Oindex = 0
            for atom in xrange(self.atomnr):
                line = self.datei.readline()
                if atom in self.atomdict[atomname]:
                    arr[frame, Oindex, :] = map(float, line.split()[1:])
                    Oindex += 1
        if verbose:
            print ""
        return frames

    def parse_OHframe(self, *args):
        Os = []
        Hs = []

        if len(args) == 2:

            Oinds = args[0]
            Hinds = args[1]

            self.datei.readline()
            self.datei.readline()

            for index in xrange(self.atomnr):
                line = self.datei.readline()
                if index in Oinds:
                    Os.append(ac.Atom(line.split()[0],map(float,line.split()[1:4]),index))
                elif index in Hinds:
                    Hs.append(ac.Atom(line.split()[0],map(float,line.split()[1:4]),index))
            self.frame += 1
            return (Os,Hs)

        else:
            self.datei.readline()
            self.datei.readline()
            for index in xrange(self.atomnr):
                line = self.datei.readline()
                if "O" in line:
                    Os.append(ac.Atom(line.split()[0],map(float,line.split()[1:4]),index))
                elif "H" in line:
                    Hs.append(ac.Atom(line.split()[0],map(float,line.split()[1:4]),index))
            self.frame += 1
            return (Os,Hs)

    def get_atoms(self, *atomnames):
        # pdb.set_trace()
        atoms = dict()
        for atomname in atomnames:
            if atomname in self.atomdict.keys():
                atoms[atomname] = []
        line = self.datei.readline()
        if line == "":
            raise EOFError
        self.datei.readline()
        for index in xrange(self.atomnr):
            line = self.datei.readline()
            for atomname in atoms.keys():
                if index in self.atomdict[atomname]:
                    atoms[atomname].append(ac.Atom(line.split()[0], map(float, line.split()[1:4]), index))
        self.frame += 1
        return atoms

    def get_atoms_numpy(self, atomnames=[]):
        # pdb.set_trace()
        if len(atomnames) == 0:
            atoms = np.zeros(self.atomnr, dtype = npa.xyzatom)
            line = self.datei.readline()
            if line == "":
                raise EOFError
            self.datei.readline()
            for index in xrange(self.atomnr):
                line = self.datei.readline()
                atoms[index]["name"] = line.split()[0]
                atoms[index]["pos"][:] = map(float, line.split()[1:4])
        else:
            atomnr = sum([len(self.atomdict[atomname]) for atomname in atomnames])
            atoms = np.zeros(atomnr, dtype = npa.xyzatom)
            line = self.datei.readline()
            if line == "":
                raise EOFError
            self.datei.readline()
            j=0
            for index in xrange(self.atomnr):
                line = self.datei.readline()
                # name = line.split()[0]
                for atomname, atom_indices in self.atomdict.iteritems():
                    if index in atom_indices:
                        atoms[j]["name"] = atomname
                        atoms[j]["pos"][:] = map(float, line.split()[1:4])
                        j += 1
        self.frame += 1

        return atoms

    def get_trajectory_numpy(self, atomnames=[], acidic_protons=False, verbose=False):
        # pdb.set_trace()
        start_time=time.time()
        traj = []
        counter = 0
        while 1:
            try:
                traj.append(self.get_atoms_numpy(atomnames))
            except EOFError:
                break
            if verbose == True and counter % 100 == 0:
                print "#Frame {}, ({:.2f} fps)".format(counter, float(counter)/(time.time()-start_time)), "\r",
            counter += 1
        if verbose:
            print ""
        if verbose:
            print "#Total time: {} sec".format(time.time() - start_time)


        return np.array(traj, dtype=npa.xyzatom)

    def print_frame(self):
        for i in xrange(self.atomnr+2):
            print self.datei.readline()[:-1]

    def print_selection(self, selection):
        atomnr = 0
        for a in self.atomdict.keys():
            if a in selection:
                atomnr += len(self.atomdict[a])
        print atomnr
        print ""
        self.datei.readline()
        self.datei.readline()
        for index in xrange(self.atomnr):
            line = self.datei.readline()
            if line.split()[0] in selection:
                print line[:-1]

    def seek_frame(self,n):
        self.datei.seek(0)
        if n == 0:
            pass
        else:
            i = 1
            while i % ((self.atomnr+2)*n + 1) != 0:
                    self.datei.readline()
                    i += 1
        self.frame = n

    def framejump(self, n):
        for i in xrange((self.atomnr + 2)*n):
            self.datei.readline()

    def get_framenumber(self, verbose=False):
        if self.framenumber != None:
            return self.framenumber
        else:
            linenumber = int(os.popen("wc -l "+self.filename).read().split()[0])
            framenumber = linenumber/(self.atomnr+2)
            if verbose:
                print "#{} frames".format(framenumber)
            return framenumber

    def rewind(self):
        self.datei.seek(0)
