#!/usr/bin/python -u
# -*- coding: utf-8 -*-
# from __future__ import print_function
import os
import sys
import time
from collections import OrderedDict
import logging
import re
import ast

import numpy as np
import ipdb
import git
import inspect
import argparse
from textwrap import wrap

from mdkmc.IO import xyz_parser
from mdkmc.IO import BinDump
from mdkmc.IO import config_parser
from mdkmc.cython_exts.kMC import kMC_helper
from mdkmc.cython_exts.atoms import numpyatom as npa

# -----------------------Determine Loglevel here----------------------------------------
# Choose between info, warning, debug
loglevel = logging.WARNING
logger = logging.getLogger(__name__)
logger.setLevel(loglevel)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# create file handler which logs even debug messages
fh = logging.FileHandler('MDMC.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
# create formatter
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
# logger.addHandler(ch)
# --------------------------------------------------------------------------------------

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print "#", os.path.join(script_path,"../cython/kMC")


class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def load_trajectory(traj, atomname, verbose=False):
    atom_number = traj[0][traj[0]["name"] == atomname].size
    atom_traj = (traj[traj["name"] == atomname]).reshape((traj.shape[0], atom_number))
    return np.array(atom_traj["pos"], order="C")


def count_atoms_in_volume(atoms, ((xmin, xmax), (ymin, ymax), (zmin, zmax)), pbc):
    # if volume.shape != (3,2):
    # 	raise ValueError("Shape of Volume needs to be (3,2). [[xmin, xmax], [ymin, ymax], [zmin, zmax]]")
    counter = 0
    indices = []
    for i, a in enumerate(atoms):
        if xmin <= a[0] % pbc[0] < xmax and ymin <= a[1] % pbc[1] < ymax and zmin <= a[2] % pbc[2] < zmax:
            counter += 1
            indices.append(i)
    return counter, indices


def count_atoms_in_volume_z(atoms, ((xmin, xmax), (ymin, ymax), (zmin, zmax)), pbc):
    counter = 0
    indices = []
    for i, a in enumerate(atoms):
        if xmin <= a[0] % pbc[0] < xmax and ymin <= a[1] % pbc[1] < ymax and zmin <= a[2] < zmax:
            counter += 1
            indices.append(i)
    return counter, indices


def print_frame(*args):
    atoms = args[0]
    if len(args) == 2:
        i = args[1]
        atomnr = atoms.shape[1]
        atom_selection = atoms[i]
    else:
        atomnr = atoms.shape[0]
        atom_selection = atoms
    print atomnr
    print ""
    for j in xrange(atomnr):
        print "{:>2} {: 20.10f} {: 20.10f} {: 20.10f}".format("O", *atom_selection[j])


def load_configfile_new(configfilename, verbose=False):
    parser_dict = config_parser.CONFIG_DICT
    config_dict = dict()
    with open(configfilename, "r") as f:
        for line in f:
            if line[0] != "#":
                if len(line.split()) > 1:
                    keyword = line.split()[0]
                    if keyword in parser_dict.keys():
                        config_dict[keyword.lower()] = parser_dict[keyword.lower()]["parse_fct"](line)
                    else:
                        raise RuntimeError("Unknown keyword {}. Please remove it.".format(keyword))

    # Check for missing options, and look if they have a default argument
    for key, value in parser_dict.iteritems():
        if key not in config_dict:
            if value["default"] == "no_default":
                raise RuntimeError("Missing value for {}".format(key))
            else:
                if verbose:
                    print "# Found no value for {} in config file".format(key)
                    print "# Will use default value: {}".format(value["default"])
                config_dict[key] = value["default"]

    return config_dict


def print_confighelp():
    text_width = 80
    parser_dict = config_parser.CONFIG_DICT
    for k, v in parser_dict.iteritems():
        keylen = len(k)
        delim_len = (text_width-2-keylen) / 2
        print "{delim} {keyword} {delim}".format(keyword=k.upper(), delim=delim_len*"-")
        print ""
        print "\n".join(wrap(v["help"], width=text_width))
        print ""
        print "Default:", v["default"]
        print text_width * "-"
        print ""
        print ""


def get_gitversion():
    repo = git.Repo(script_path)
    print "# Hello. I am from commit {}".format(repo.active_branch.commit.hexsha)


def count_protons_and_oxygens(Opos, proton_lattice, O_counter, H_counter, bin_bounds):
    for i in xrange(proton_lattice.shape[0]):
        O_z = Opos[i, 2]
        O_index = np.searchsorted(bin_bounds, O_z)-1
        O_counter[O_index] += 1
        if proton_lattice[i] > 0:
            H_counter[O_index] += 1


def remove_com_movement(trajectory, verbose):
    if verbose:
        print "#removing COM movement"
    for i, frame in enumerate(trajectory):
        if verbose is True and i % 1000 == 0:
            print "#Frame", i,
            print "\r",
        com = frame.sum(axis=0)/float(frame.shape[0])
        frame -= com
    if verbose:
        print ""


def remove_com_movement_frame(frame):
    com = frame.sum(axis=0)/float(frame.shape[0])
    frame -= com


def extend_simulationbox(Opos, onumber, h, box_multiplier, nonortho=False):
    if True in [multiplier > 1 for multiplier in box_multiplier]:
        if nonortho:
            v1 = h[:, 0]
            v2 = h[:, 1]
            v3 = h[:, 2]
            Oview = Opos.view()
            Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], onumber, 3)
            for x in xrange(box_multiplier[0]):
                for y in xrange(box_multiplier[1]):
                    for z in xrange(box_multiplier[2]):
                        if x+y+z != 0:
                            for i in xrange(onumber):
                                Oview[x, y, z, i, :] = Oview[0, 0, 0, i] + x * v1 + y * v2 + z * v3
        else:
            ipdb.set_trace()
            Oview = Opos.view()
            Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], onumber, 3)
            kMC_helper.extend_simulationbox(Oview, h,
                                            box_multiplier)


def calculate_displacement(proton_lattice, proton_pos_snapshot,
                           Opos_new, displacement, pbc, wrap=True):
    # ipdb.set_trace()
    proton_pos_new = np.zeros(proton_pos_snapshot.shape)
    for O_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index-1] = Opos_new[O_index]
    if wrap:
        kMC_helper.dist_numpy_all_inplace(displacement, proton_pos_new,
                                          proton_pos_snapshot, pbc)
    else:
        displacement += kMC_helper.dist_numpy_all(proton_pos_new, proton_pos_snapshot, pbc)
        proton_pos_snapshot[:] = proton_pos_new   # [:] is important (inplace operation)


def calculate_displacement_nonortho(proton_lattice, proton_pos_snapshot,
                                    Opos_new, displacement,
                                    h, h_inv):
    # ipdb.set_trace()
    proton_pos_new = np.zeros(proton_pos_snapshot.shape)
    for O_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index-1] = Opos_new[O_index]
    kMC_helper.dist_numpy_all_nonortho(displacement, proton_pos_new, proton_pos_snapshot, h, h_inv)


def calculate_autocorrelation(protonlattice_old, protonlattice_new):
    autocorrelation = 0
    for i in xrange(protonlattice_new.size):
        if protonlattice_old[i] == protonlattice_new[i] != 0:
            autocorrelation += 1
    return autocorrelation


def calculate_MSD(MSD, displacement):
    MSD *= 0
    for d in displacement:
        MSD += d*d
    MSD /= displacement.shape[0]


class MDMC:
    def __init__(self, **kwargs):
        # Create Loggers
        self.logger = logging.getLogger("{}.{}".format(__name__, self.__class__.__name__))
        self.logger.info("Creating an instance of {}.{}".format(__name__, self.__class__.__name__))
        try:
            get_gitversion()
        except git.InvalidGitRepositoryError:
            print "# No Git repository found, so I cannot show any commit information"
        if "configfile" in kwargs.keys():
            # Load settings from config file
            file_kwargs = load_configfile_new(kwargs["configfile"], verbose=True)
            if ("verbose", True) in file_kwargs.viewitems():
                print "# Config file specified. Loading settings from there."

            # Save settings as object variable
            self.__dict__.update(file_kwargs)

            self.trajectory = BinDump.npload_atoms(self.filename, create_if_not_existing=True,
                                                   remove_com=True, verbose=self.verbose)

            if self.clip_trajectory is not None:
                if self.clip_trajectory >= self.trajectory.shape[0]:
                    print "# Trying to clip trajectory to {} frames, but trajectory only has {} frames!"\
                        .format(self.clip_trajectory, self.trajectory.shape[0])
                else:
                    if self.verbose:
                        print "# Clipping trajectory from frame 0 to frame {}".format(self.clip_trajectory)
                    self.trajectory = self.trajectory[:self.clip_trajectory]

            self.O_trajectory = load_trajectory(self.trajectory, "O", verbose=self.verbose)
            if self.po_angle:
                self.P_trajectory = load_trajectory(self.trajectory, self.o_neighbor, verbose=self.verbose)
        else:
            pass

    def determine_PO_pairs(self, framenumber, atombox):
        P_neighbors = np.zeros(self.oxygennumber_extended, int)
        Os = atombox.get_extended_frame(atombox.oxygen_trajectory[framenumber])
        Ps = atombox.get_extended_frame(atombox.phosphorus_trajectory[framenumber])

        if self.nonortho:
            for i in xrange(Os.shape[0]):
                P_index = npa.nextNeighbor_nonortho(Os[i], Ps, self.h, self.h_inv)[0]
                P_neighbors[i] = P_index
        else:
            for i in xrange(Os.shape[0]):
                P_index = npa.nextNeighbor(Os[i], Ps, atombox.periodic_boundaries_extended)[0]
                P_neighbors[i] = P_index
        return P_neighbors

    def determine_PO_angles(self, O_frame, P_frame, P_neighbors, angles):
        if self.nonortho:
            for i in xrange(O_frame.shape[0]):
                for j in xrange(i):
                    if i != j and npa.length_nonortho_bruteforce(O_frame[i], O_frame[j], self.h, self.h_inv) < self.cutoff_radius:
                        angles[i, j] = npa.angle_nonortho(O_frame[i], P_frame[P_neighbors[i]], O_frame[j], P_frame[P_neighbors[j]], self.h, self.h_inv)
        else:
            for i in xrange(O_frame.shape[0]):
                for j in xrange(i):
                    if i != j and npa.sqdist(O_frame[i], O_frame[j], self.pbc) < self.cutoff_radius:
                        PO1 = npa.distance(O_frame[i], P_frame[P_neighbors[i]], self.pbc)
                        PO2 = npa.distance(O_frame[j], P_frame[P_neighbors[j]], self.pbc)
                        angles[i,j] = np.dot(PO1, PO2)/np.linalg.norm(PO1)/np.linalg.norm(PO2)

#TODO print_settings fertig!
    def print_settings(self):
        print "# I'm using the following settings:"
        for k, v in self.__dict__.iteritems():
            if "trajectory" in k:
                pass
            elif k == "angles":
                pass
            elif k == "P_neighbors":
                pass
            elif k == "default_dict":
                pass
            elif k == "h":
                print "# h = {} {} {}".format(*v[0])
                print "#     {} {} {}".format(*v[1])
                print "#     {} {} {}".format(*v[2])
            elif k == "h_inv":
                print "# h_inv = {} {} {}".format(*v[0])
                print "#         {} {} {}".format(*v[1])
                print "#         {} {} {}".format(*v[2])
            else:
                print "# {:20} {:>20}".format(k, v)

    def init_proton_lattice(self, box_multiplier):
        proton_lattice = np.zeros(self.oxygennumber*box_multiplier[0]*box_multiplier[1]*box_multiplier[2], np.uint8)
        proton_lattice[:self.proton_number] = xrange(1, self.proton_number+1)
        np.random.shuffle(proton_lattice)

        return proton_lattice

    def init_observables_protons_constant(self):
        displacement = np.zeros((self.proton_number, 3))
        MSD = np.zeros(3)
        proton_pos_snapshot = np.zeros((self.proton_number, 3))
        proton_pos_new = np.zeros((self.proton_number, 3))

        return displacement, MSD, proton_pos_snapshot, proton_pos_new

    def init_observables_protons_variable(self, O_trajectory, zfac, bins):
        r = np.zeros(3, np.float32)
        zmin = O_trajectory[:, :, 2].min()
        zmax = O_trajectory[:, :, 2].max()
        zmax += (zfac-1)*self.pbc[2]

        O_counter = np.zeros(bins, int)
        H_counter = np.zeros(bins, int)
        bin_bounds = np.linspace(zmin, zmax, bins+1)

        return r, O_counter, H_counter, bin_bounds

    def reset_observables(self, proton_pos_snapshot, protonlattice, displacement, Opos, helper):
        for i, site in enumerate(protonlattice):
            if site > 0:
                proton_pos_snapshot[site-1] = Opos[i]

        protonlattice_snapshot = np.copy(protonlattice)

        helper.reset_jumpcounter()

        if not self.periodic_wrap:
            displacement[:] = 0

        return protonlattice_snapshot, proton_pos_snapshot, displacement

    def print_observable_names(self):
        print >> self.output, "# {:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10} {:>12}".format(
            "Sweeps", "Time", "MSD_x", "MSD_y", "MSD_z", "Autocorr", "Jumps", "Sweeps/Sec", "Remaining Time/Min")

    def print_observables(self, sweep, autocorrelation, helper, timestep_fs, start_time, MSD):
        speed = float(sweep)/(time.time()-start_time)
        if speed != 0:
            remaining_time = (self.sweeps - sweep)/speed/60
        else:
            remaining_time = -1
        print >> self.output, " {:>10} {:>10}    {:18.8f} {:18.8f} {:18.8f} {:8d} {:10d} {:10.2f} {:10.2f}".format(
            sweep, sweep*timestep_fs, MSD[0], MSD[1], MSD[2], autocorrelation, helper.get_jumps(), speed, remaining_time)

    def print_observables_reservoir(self, sweep, timestep_fs, r, jumps, start_time):
        print >> self.output, " {:>10} {:>10}    {:18.8f} {:18.8f} {:18.8f} {:8d} {:10.2f}".format(
            sweep, sweep*timestep_fs, r[0], r[1], r[2], jumps, float(sweep)/(time.time()-start_time))

    def print_OsHs(self, Os, proton_lattice, sweep, timestep_fs):
        proton_indices = np.where(proton_lattice > 0)[0]
        print Os.shape[0] + self.proton_number
        print "Time:", sweep*timestep_fs
        for i in xrange(Os.shape[0]):
            print "O		{:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i])
        for index in proton_indices:
            print "H		{:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index])

    def extend_simulationbox_old(self, Opos, onumber, h, box_multiplier, pbc):
        if True in [multiplier > 1 for multiplier in box_multiplier]:
            if self.nonortho:
                v1 = h[:, 0]
                v2 = h[:, 1]
                v3 = h[:, 2]
                Oview = Opos.view()
                Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], onumber, 3)
                for x in xrange(box_multiplier[0]):
                    for y in xrange(box_multiplier[1]):
                        for z in xrange(box_multiplier[2]):
                            if x+y+z != 0:
                                for i in xrange(onumber):
                                    Oview[x, y, z, i, :] = Oview[0, 0, 0, i] + x * v1 + y * v2 + z * v3
            else:
                Oview = Opos.view()
                Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], onumber, 3)
                for x in xrange(box_multiplier[0]):
                    for y in xrange(box_multiplier[1]):
                        for z in xrange(box_multiplier[2]):
                            if x+y+z != 0:
                                for i in xrange(onumber):
                                    Oview[x, y, z, i, :] = Oview[0, 0, 0, i] + pbc * [x, y, z]

    def get_average_transitionmatrix_reservoir(self, helper, O_trajectory,
                                               parameters, timestep, pbc,
                                               nointra=True, verbose=False):
        if os.path.exists(self.trajectory+"tm_avg.npy"):
            if verbose:
                print "#average matrix already on disk"
                print "#loading from", self.trajectory+"tm_avg.npy"
            tm_avg = np.load(self.trajectory+"tm_avg.npy")
            print "#",tm_avg.shape
        else:
            tm_avg = np.zeros((722, 722))
            Opos = np.zeros((5*self.oxygennumber, 3), np.float32)

            if verbose:
                print "#calculating new average matrix"

            for i in xrange(O_trajectory.shape[0]):
                Opos[:144,:] = O_trajectory[i]
                kMC_helper.extend_simulationbox_z(5, Opos, self.pbc[2], 144)
        # 		ipdb.set_trace()
                if i % 1000 == 0:
                    print "#{}/{} frames".format(i, O_trajectory.shape[0]),
                    print "\r",
                helper.calculate_transitionsmatrix_reservoir(tm_avg, Opos, parameters, self.pbc, 4.0, 6.0)
            tm_avg /= O_trajectory.shape[0]
            np.save(self.trajectory+"tm_avg.npy", tm_avg)
            print "\n#done"
        return tm_avg

    def ASEP(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(2**32)
            np.random.seed(self.seed)
            #~ print "#Using seed {}".format(self.seed)
        self.oxygennumber = self.O_trajectory.shape[1]
        self.oxygennumber_extended = self.O_trajectory.shape[1] * self.box_multiplier[0] * self.box_multiplier[1] * self.box_multiplier[2]

        if "A" in self.jumprate_params_fs.keys():
            self.jumprate_params_fs["A"] *= self.md_timestep_fs
        else:
            self.jumprate_params_fs["a"] *= self.md_timestep_fs
        proton_lattice = self.init_proton_lattice(self.box_multiplier)
        #Check PBC and determine whether cell is orthorhombic or non-orthorhombic
        if len(self.pbc) == 3:
            self.nonortho = False
            if self.po_angle:
                atombox = kMC_helper.AtomBox_Cubic(self.O_trajectory, self.pbc,
                                                   np.array(self.box_multiplier, dtype=np.int32),
                                                   self.P_trajectory)
            else:
                atombox = kMC_helper.AtomBox_Cubic(self.O_trajectory, self.pbc,
                                                   np.array(self.box_multiplier, dtype=np.int32))
        else:
            self.nonortho = True
            if self.po_angle:
                atombox = kMC_helper.AtomBox_Monoclin(self.O_trajectory, self.pbc,
                                                   np.array(self.box_multiplier, dtype=np.int32),
                                                   self.P_trajectory)
            else:
                atombox = kMC_helper.AtomBox_Monoclin(self.O_trajectory, self.pbc,
                                                   np.array(self.box_multiplier, dtype=np.int32))

        displacement, MSD, proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant()
        #only update neighborlists after neighbor_freq position updates
        neighbor_update = self.neighbor_freq*(self.skip_frames+1)

        if self.po_angle:
            self.phosphorusnumber = self.P_trajectory.shape[1]
            self.phosphorusnumber_extended = self.phosphorusnumber * self.box_multiplier[0] * self.box_multiplier[1] *self.box_multiplier[2]
            self.P_neighbors = self.determine_PO_pairs(framenumber=0, atombox=atombox)
            self.angles = np.zeros((self.oxygennumber_extended, self.oxygennumber_extended))
            #~ self.determine_PO_angles(self.O_trajectory[0], self.P_trajectory[0], self.P_neighbors, self.angles)

        if self.verbose:
            print >> self.output, "# Sweeps:", self.sweeps
        self.print_settings()

        Opos = np.zeros((self.box_multiplier[0]*self.box_multiplier[1]*self.box_multiplier[2]*self.oxygennumber, 3), np.float64)
        if self.po_angle:
            Ppos = np.zeros((self.box_multiplier[0]*self.box_multiplier[1]*self.box_multiplier[2]*self.phosphorusnumber, 3), np.float64)

        # transitionmatrix = np.zeros((Opos.shape[0], Opos.shape[0]))
        # jumpmatrix = np.zeros((Opos.shape[0], Opos.shape[0]), int)

        start_time = time.time()

        helper = kMC_helper.Helper(atombox=atombox, pbc=self.pbc,
                                   box_multiplier=np.array(self.box_multiplier, dtype=np.int32),
                                   P_neighbors=np.array(self.P_neighbors, dtype=np.int32),
                                   nonortho=self.nonortho,
                                   jumprate_parameter_dict=self.jumprate_params_fs,
                                   cutoff_radius=self.cutoff_radius,
                                   angle_threshold=self.angle_threshold,
                                   neighbor_search_radius=self.neighbor_search_radius,
                                   jumprate_type=self.jumprate_type)

        # Equilibration
        for sweep in xrange(self.equilibration_sweeps):
            if sweep % 1000 == 0:
                print >> self.output, "# Equilibration sweep {}/{}".format(sweep, self.equilibration_sweeps, ), "\r",
            if sweep % (self.skip_frames+1) == 0:
                if not self.shuffle:
                    frame = sweep
                #     Opos[:self.oxygennumber] = self.O_trajectory[sweep % self.O_trajectory.shape[0]]
                else:
                    frame = np.random.randint(self.O_trajectory.shape[0])
                #     Opos[:self.oxygennumber] = self.O_trajectory[np.random.randint(self.O_trajectory.shape[0])]
                # extend_simulationbox(Opos, self.oxygennumber, self.h, self.box_multiplier)
                # if self.po_angle:
                #     Ppos[:self.phosphorusnumber] = self.P_trajectory[sweep % self.P_trajectory.shape[0]]
                #     extend_simulationbox(Ppos, self.phosphorusnumber, self.h, self.box_multiplier)
                if sweep % neighbor_update == 0:
                    helper.determine_neighbors(frame % self.O_trajectory.shape[0])
                #     print helper.neighbors
                    # if self.po_angle:
                    #     helper.calculate_transitions_POOangle(Opos, Ppos, self.P_neighbors, self.cutoff_radius, self.angle_threshold)
                    # else:
                    #     helper.calculate_transitions_new(Opos, self.jumprate_params_fs, self.cutoff_radius)

            helper.sweep_from_vector(sweep % self.O_trajectory.shape[0], proton_lattice)

        if not self.xyz_output:
            self.print_observable_names()

        #~ print >> self.output, "#{:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10} {:>12}".format("Sweeps", "Time", "MSD_x", "MSD_y", "MSD_z", "Autocorr", "Jumps", "Sweeps/Sec", "Remaining Time/Min")
        #Run
        for sweep in xrange(0, self.sweeps):
            if sweep % (self.skip_frames+1) == 0:
                if not self.shuffle:
                    frame = sweep % self.O_trajectory.shape[0]
                #     Opos[:self.oxygennumber] = self.O_trajectory[sweep%self.O_trajectory.shape[0]]
                else:
                    frame = np.random.randint(self.O_trajectory.shape[0])
                #     Opos[:self.oxygennumber] = self.O_trajectory[np.random.randint(self.O_trajectory.shape[0])]
                # extend_simulationbox(Opos, self.oxygennumber, self.h, self.box_multiplier)
                # if self.po_angle:
                #     Ppos[:self.phosphorusnumber] = self.P_trajectory[sweep%self.P_trajectory.shape[0]]
                #     extend_simulationbox(Ppos, self.phosphorusnumber, self.h, self.box_multiplier)
                if sweep % neighbor_update == 0:
                    helper.determine_neighbors(frame)
                # if self.po_angle:
                #     helper.calculate_transitions_POOangle(Opos, Ppos, self.P_neighbors, self.cutoff_radius, self.angle_threshold)
                # else:
                #     helper.calculate_transitions_new(Opos, self.jumprate_params_fs, self.cutoff_radius)
            helper.sweep_from_vector(frame, proton_lattice)

            if sweep % self.reset_freq == 0:
                protonlattice_snapshot, proton_pos_snapshot, displacement = \
                    self.reset_observables(proton_pos_snapshot, proton_lattice, displacement,
                                           atombox.get_extended_frame(atombox.oxygen_trajectory[frame]), helper)
            if  sweep % self.print_freq == 0:
                # self.remove_com_movement_frame(Opos)
                if not self.nonortho:
                    # print " ".join(map(lambda x: "{:02d}".format(x), proton_lattice))
                    # print "proton lattice:", len(proton_lattice)
                    calculate_displacement(proton_lattice, proton_pos_snapshot,
                                           atombox.get_extended_frame(atombox.oxygen_trajectory[frame]),
                                           displacement, self.pbc, wrap=self.periodic_wrap)
                else:
                    calculate_displacement_nonortho(proton_lattice, proton_pos_snapshot, proton_pos_new, helper.oxygen_frame_extended, displacement, self.h, self.h_inv)

                calculate_MSD(MSD, displacement)
                autocorrelation = calculate_autocorrelation(protonlattice_snapshot, proton_lattice)
                if not self.xyz_output:
                    self.print_observables(sweep, autocorrelation, helper, self.md_timestep_fs, start_time, MSD)
                else:
                    self.print_OsHs(Opos, proton_lattice, sweep, self.md_timestep_fs)
            # helper.sweep_list(proton_lattice)
            if self.jumpmatrix:
                helper.sweep_from_vector_jumpmatrix(sweep % self.O_trajectory.shape[0], proton_lattice)
            else:
                helper.sweep_from_vector(sweep % self.O_trajectory.shape[0], proton_lattice)

            # jumps = helper.sweep_gsl(proton_lattice, transitionmatrix)
            #~ jumps = helper.sweep_list_jumpmat(proton_lattice, jumpmatrix)

        #~ return jumpmatrix

def main(*args):
    parser=argparse.ArgumentParser(description="MDMC Test", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configfile", help="config file")
    parser.add_argument("--confighelp", "-c", action="store_true", help="config file help")
    args = parser.parse_args()

    if args.confighelp:
        print_confighelp()
        sys.exit(0)

    md_mc = MDMC(configfile=args.configfile)
    # md_mc.print_settings()
    start_time = time.time()
    md_mc.ASEP()

    print "#Total time: {:.1f} minutes".format((time.time()-start_time)/60)

if __name__ == "__main__":
    main()
