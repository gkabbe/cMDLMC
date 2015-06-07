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
# import ipdb
import git
import inspect
import argparse

from mdkmc.IO import xyzparser
from mdkmc.IO import BinDump
from mdkmc.cython_exts.kMC import kMC_helper
from mdkmc.cython_exts.atoms import numpyatom as npa

# -----------------------Determine Loglevel here----------------------------------------
# Choose between info, warning, debug
loglevel = logging.INFO
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
#~ logger.addHandler(ch)
#--------------------------------------------------------------------------------------

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#~ print "#", os.path.join(script_path,"../cython/kMC")


class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def load_trajectory(traj, atomname, verbose=False):
    #~ traj = traj = BinDump.npload_atoms(traj_fname, create_if_not_existing=True, verbose=verbose)
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


def load_configfile(configfilename):
    def get_jumprate_parameters(line):
        dict_string = re.findall("\{.*\}|dict\s*\(.*\)", line)[0]
        print "param dict:", dict_string
        param_dict = eval(dict_string)
        return param_dict

    def parse_int(line):
        return int(line.split()[1])

    def parse_float(line):
        return float(line.split()[1])

    def string2bool(s):
        if s.upper() == "TRUE":
            return True
        elif s.upper() == "FALSE":
            return False
        else:
            print s
            raise(ValueError("Unknown value. Please use True or False"))

    parser_dict = dict()
    parser_dict["filename"]             = lambda line: line.split()[1]
    parser_dict["output"]               = lambda line: open(line.split()[1], "w")
    parser_dict["o_neighbor"]           = lambda line: line.split()[1].upper()
    parser_dict["jumprate_type"]        = lambda line: line.split()[1]
    parser_dict["sweeps"]               = parse_int
    parser_dict["equilibration_sweeps"] = parse_int
    parser_dict["skip_frames"]          = parse_int
    parser_dict["print_freq"]           = parse_int
    parser_dict["reset_freq"]           = parse_int
    parser_dict["neighbor_freq"]        = parse_int
    parser_dict["framenumber"]          = parse_int
    parser_dict["proton_number"]        = parse_int
    parser_dict["clip_trajectory"]      = parse_int
    parser_dict["seed"]                 = parse_int
    parser_dict["md_timestep_fs"]       = parse_float
    #~ parser_dict["threshold"]            = parse_float
    parser_dict["angle_threshold"]      = parse_float
    parser_dict["cutoff_radius"]        = parse_float
    #~ parser_dict["dump_trajectory"]      = lambda line: string2bool(line.split()[1])
    parser_dict["po_angle"]             = lambda line: string2bool(line.split()[1])
    parser_dict["shuffle"]              = lambda line: string2bool(line.split()[1])
    parser_dict["verbose"]              = lambda line: string2bool(line.split()[1])
    parser_dict["xyz_output"]           = lambda line: string2bool(line.split()[1])
    parser_dict["box_multiplier"]       = lambda line: map(int, line.split()[1:])
    parser_dict["pbc"]                  = lambda line: np.array(map(float, line.split()[1:]))
    parser_dict["jumprate_params_fs"]   = get_jumprate_parameters

    config_dict = dict()
    with open(configfilename, "r") as f:
        for line in f:
            if line[0] != "#":
                if len(line.split()) > 1 and line.split()[0] in parser_dict.keys():
                    config_dict[line.split()[0].lower()] = parser_dict[line.split()[0].lower()](line)

    return config_dict

def create_default_dict():
    default_dict = OrderedDict()

    default_dict["filename"]             = "no_default"
    default_dict["equilibration_sweeps"] = "no_default"
    default_dict["skip_frames"]          = "no_default"
    default_dict["print_freq"]           = "no_default"
    default_dict["reset_freq"]           = "no_default"
    default_dict["neighbor_freq"]        = "no_default"
    default_dict["proton_number"]        = "no_default"
    default_dict["md_timestep_fs"]       = "no_default"
    default_dict["pbc"]                  = "no_default"
    default_dict["jumprate_params_fs"]   = "no_default"
    default_dict["jumprate_type"]         = "no_default"

    default_dict["cutoff_radius"]        = 4.0
    default_dict["box_multiplier"]       = [1,1,1]
    default_dict["framenumber"]          = None
    default_dict["verbose"]              = False
    default_dict["po_angle"]             = False
    default_dict["shuffle"]              = False
    default_dict["xyz_output"]           = False
    default_dict["output"]               = sys.stdout
    default_dict["angle_threshold"]      = 1.57
    default_dict["o_neighbor"]           = "P"
    default_dict["sweeps"]               = "no_default"
    default_dict["clip_trajectory"]      = None
    default_dict["seed"]                 = None

    return default_dict

#~ def set_default_values(config_dict, verbose=False):
    #~ def check_and_set(key, conf_dict, default_value, verbose):
        #~ if not key in config_dict.keys():
            #~ if verbose == True:
                #~ print "No specification of {} found. Using default {}".format(key, default_value)
            #~ config_dict[key] = default_value
        #~ elif verbose == True:
            #~ print "Value of {} was specified as {}".format(key, config_dict[key])
        #~
    #~ check_and_set("box_multiplier", config_dict, [1,1,1], verbose)
    #~ check_and_set("framenumber", config_dict, None, verbose)
    #~ check_and_set("verbose", config_dict, False, verbose)
    #~ check_and_set("po_angle", config_dict, False, verbose)
    #~ check_and_set("output", config_dict, sys.stdout, verbose)
    #~ check_and_set("shuffle", config_dict, False, verbose)

class MDMC:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger("{}.{}".format(__name__, self.__class__.__name__))
        self.logger.info("Creating an instance of {}.{}".format(__name__, self.__class__.__name__))
        self.get_gitversion()
        self.default_dict = create_default_dict()
        if "configfile" in kwargs.keys():
            file_kwargs = load_configfile(kwargs["configfile"])
            if ("verbose", True) in file_kwargs.viewitems():
                print "#Config file specified. Loading settings from there."
            self.check_arguments(**file_kwargs)

            self.trajectory = BinDump.npload_atoms(self.filename, create_if_not_existing=True, remove_com=True, verbose=self.verbose)

            if self.clip_trajectory is not None:
                if self.clip_trajectory >= self.trajectory.shape[0]:
                    print "#Trying to clip trajectory to {} frames, but trajectory only has {} frames!".format(self.clip_trajectory, self.trajectory.shape[0])
                else:
                    if self.verbose:
                        print "#Clipping trajectory from frame 0 to frame {}".format(self.clip_trajectory)
                    self.trajectory = self.trajectory[:self.clip_trajectory]

            self.O_trajectory = load_trajectory(self.trajectory, "O", verbose=self.verbose)
        else:
            pass


    def get_gitversion(self):
        repo = git.Repo(script_path)
        print "#Hello. I am from commit {}".format(repo.active_branch.commit.hexsha)


    def check_arguments(self, **kwargs):
        def check_if_correct(key, value):
            if key == "pbc":
                if len(value) not in [3, 9]:
                    raise InputError("Wrong input length for {}: {}".format(key, value))

        for arg in self.default_dict.keys():
            try:
                check_if_correct(arg, kwargs[arg])
                self.__dict__[arg] = kwargs[arg]
            except KeyError:
                if self.default_dict[arg] != "no_default":
                    if ("verbose", True) in kwargs.viewitems():
                        print "#Missing value for argument {}".format(arg)
                        print "#Using default value {}".format(self.default_dict[arg])
                    self.__dict__[arg] = self.default_dict[arg]
                else:
                    if ("verbose", True) in kwargs.viewitems():
                        print "#No value specified for {}, no default value found.".format(arg)
                        print "#Exiting"
                        sys.exit(1)

        for key in kwargs.keys():
            if key not in self.default_dict.keys() and key != "default_dict":
                print "#Ignoring unknown argument {}".format(key)


    def determine_PO_pairs(self):
        P_neighbors = np.zeros(self.oxygennumber_extended, int)
        Os = np.zeros((self.oxygennumber_extended, 3))
        Ps = np.zeros((self.phosphorusnumber_extended, 3))
        Os[:self.oxygennumber] = self.O_trajectory[0]
        Ps[:self.phosphorusnumber] = self.P_trajectory[0]
        self.extend_simulationbox(Ps, self.phosphorusnumber, self.h, self.box_multiplier)
        self.extend_simulationbox(Os, self.oxygennumber, self.h, self.box_multiplier)

        if self.nonortho:
            for i in xrange(Os.shape[0]):
                P_index = npa.nextNeighbor_nonortho(Os[i], Ps, self.h, self.h_inv)[0]
                P_neighbors[i] = P_index
        else:
            for i in xrange(Os.shape[0]):
                P_index = npa.nextNeighbor(Os[i], Ps, self.pbc_extended)[0]
                P_neighbors[i] = P_index
        return P_neighbors

    def determine_PO_angles(self, O_frame, P_frame, P_neighbors, angles):
        if self.nonortho:
            for i in xrange(O_frame.shape[0]):
                for j in xrange(i):
                    if i != j and npa.length_nonortho_bruteforce(O_frame[i], O_frame[j], self.h, self.h_inv) < self.cutoff_radius:
                        angles[i,j] = npa.angle_nonortho(O_frame[i], P_frame[P_neighbors[i]], O_frame[j], P_frame[P_neighbors[j]], self.h, self.h_inv)
        else:
            for i in xrange(O_frame.shape[0]):
                for j in xrange(i):
                    if i != j and npa.sqdist(O_frame[i], O_frame[j], self.pbc) < self.cutoff_radius:
                        PO1 = npa.distance(O_frame[i], P_frame[P_neighbors[i]], self.pbc)
                        PO2 = npa.distance(O_frame[j], P_frame[P_neighbors[j]], self.pbc)
                        angles[i,j] = np.dot(PO1, PO2)/np.linalg.norm(PO1)/np.linalg.norm(PO2)

#TODO print_settings fertig!
    def print_settings(self):
        print "#I'm using the following settings:"
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
                print "#{:20} {:>20}".format(k, v)
            #~ elif k == "default_dict":
                #~ print "#Default Dict Values:"
                #~ for key, value in v.iteritems():
                    #~ print "#", key, value
    #~ def print_settings(self, sweeps, equilibration_sweeps, MD_timestep_fs, skip_frames, print_freq, reset_freq, dump_trajectory, box_multiplier, neighbor_freq, verbose=False):
        #~ print "#Parameters used for jump rate function : {} (fs^-1) {} (Angstroem) {}".format(*self.jumprate_params_fs)
        #~ print "#Total number of sweeps: {}, reset of MSD and autocorrelation after {} sweeps".format()

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
        zmin = O_trajectory[:, :,2].min()
        zmax = O_trajectory[:, :, 2].max()
        zmax += (zfac-1)*self.pbc[2]

        O_counter = np.zeros(bins, int)
        H_counter = np.zeros(bins, int)
        bin_bounds = np.linspace(zmin, zmax, bins+1)

        return r, O_counter, H_counter, bin_bounds

    def count_protons_and_oxygens(self, Opos, proton_lattice, O_counter, H_counter, bin_bounds):
        for i in xrange(proton_lattice.shape[0]):
            O_z = Opos[i,2]
            O_index = np.searchsorted(bin_bounds, O_z)-1
            O_counter[O_index] += 1
            if proton_lattice[i] > 0:
                H_counter[O_index] += 1

#~
    #~ def read_trajectory(self, frames, oxygennumber, verbose=False):
        #~ if verbose == True:
            #~ print "#Reading trajectory..."
#~
        #~ self.xyz_file.rewind()
        #~ O_trajectory = np.zeros((frames, oxygennumber, 3), np.float32)
        #~ self.xyz_file.parse_frames_np_inplace(O_trajectory, "O", verbose)
#~
        #~ self.remove_com_movement(O_trajectory, verbose)
#~
        #~ if verbose == True:
            #~ print "#Shape:", O_trajectory.shape
        #~ return O_trajectory

    def remove_com_movement(self, trajectory, verbose):
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

    def remove_com_movement_frame(self, frame):
        com = frame.sum(axis=0)/float(frame.shape[0])
        frame -= com

    def reset_observables(self, proton_pos_snapshot, protonlattice, Opos, helper):
        for i, site in enumerate(protonlattice):
            if site > 0:
                proton_pos_snapshot[site-1] = Opos[i]

        protonlattice_snapshot = np.copy(protonlattice)

        helper.reset_jumpcounter()

        return protonlattice_snapshot, proton_pos_snapshot

    def print_observable_names(self):
        print >> self.output, "#{:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10} {:>12}".format(
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

    def extend_simulationbox(self, Opos, onumber, h, box_multiplier, nonortho=False):
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
                Oview = Opos.view()
                Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], onumber, 3)
                kMC_helper.extend_simulationbox(Oview, h,
                                                box_multiplier[0],
                                                box_multiplier[1],
                                                box_multiplier[2],
                                                onumber)

    def calculate_displacement(self, proton_lattice, proton_pos_snapshot, proton_pos_new, Opos_new, displacement, pbc):
        # ipdb.set_trace()
        for O_index, proton_index in enumerate(proton_lattice):
            if proton_index > 0:
                proton_pos_new[proton_index-1] = Opos_new[O_index]
            kMC_helper.dist_numpy_all(displacement, proton_pos_new, proton_pos_snapshot, self.pbc)

    def calculate_displacement_nonortho(self, proton_lattice, proton_pos_snapshot,
                                        proton_pos_new, Opos_new, displacement,
                                        h, h_inv):
        # ipdb.set_trace()
        for O_index, proton_index in enumerate(proton_lattice):
            if proton_index > 0:
                proton_pos_new[proton_index-1] = Opos_new[O_index]
            kMC_helper.dist_numpy_all_nonortho(displacement, proton_pos_new, proton_pos_snapshot, h, h_inv)

    def calculate_autocorrelation(self, protonlattice_old, protonlattice_new):
        autocorrelation = 0
        for i in xrange(protonlattice_new.size):
            if protonlattice_old[i] == protonlattice_new[i] != 0:
                autocorrelation += 1
        return autocorrelation

    def calculate_MSD(self, MSD, displacement):
        MSD *= 0
        for d in displacement:
            MSD += d*d
        MSD /= displacement.shape[0]

    def calculate_transitionmatrix(self, transitionmatrix, atom_positions,
                                   parameters, timestep, pbc, nointra=True):
        # ipdb.set_trace()
        for i in xrange(atom_positions.shape[0]):
            for j in xrange(atom_positions.shape[0]):
                if i != j and (nointra is False or (i/3 != j/3)):
                    transitionmatrix[i, j] = fermi(
                        parameters[0], parameters[1], parameters[2],
                        kMC_helper.dist(atom_positions[i], atom_positions[j], pbc))

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
            self.seed = np.random.randint(np.iinfo(np.int).max)
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
            self.pbc_extended = self.pbc * self.box_multiplier
            self.h = np.zeros((3,3))
            self.h[0,0] = self.pbc[0]
            self.h[1,1] = self.pbc[1]
            self.h[2,2] = self.pbc[2]
        else:
            self.pbc_extended = np.copy(self.pbc)
            self.pbc_extended[0:3] *= self.box_multiplier[0]
            self.pbc_extended[3:6] *= self.box_multiplier[1]
            self.pbc_extended[6:9] *= self.box_multiplier[2]
            self.nonortho = True
            self.h = np.array(self.pbc.reshape((3,3)).T, order="C")
            self.h_inv = np.array(np.linalg.inv(self.h), order="C")
            self.h_ext = np.array(self.pbc_extended.reshape((3,3)).T, order="C")
            self.h__ext_inv = np.array(np.linalg.inv(self.h_ext), order="C")
        displacement, MSD, proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant()
        #only update neighborlists after neighbor_freq position updates
        neighbor_update = self.neighbor_freq*(self.skip_frames+1)

        if self.po_angle:
            self.P_trajectory = load_trajectory(self.trajectory, self.o_neighbor, verbose=self.verbose)
            self.phosphorusnumber = self.P_trajectory.shape[1]
            self.phosphorusnumber_extended = self.phosphorusnumber * self.box_multiplier[0] * self.box_multiplier[1] *self.box_multiplier[2]
            self.P_neighbors = self.determine_PO_pairs()
            self.angles = np.zeros((self.oxygennumber_extended, self.oxygennumber_extended))
            #~ self.determine_PO_angles(self.O_trajectory[0], self.P_trajectory[0], self.P_neighbors, self.angles)

        if self.verbose:
            print >> self.output, "#sweeps:", self.sweeps
        self.print_settings()

        Opos = np.zeros((self.box_multiplier[0]*self.box_multiplier[1]*self.box_multiplier[2]*self.oxygennumber, 3), np.float64)
        if self.po_angle:
            Ppos = np.zeros((self.box_multiplier[0]*self.box_multiplier[1]*self.box_multiplier[2]*self.phosphorusnumber, 3), np.float64)

        transitionmatrix = np.zeros((Opos.shape[0], Opos.shape[0]))
        jumpmatrix = np.zeros((Opos.shape[0], Opos.shape[0]), int)

        start_time = time.time()

        helper = kMC_helper.Helper(self.pbc_extended, self.nonortho,
                                   jumprate_parameter_dict=self.jumprate_params_fs,
                                   jumprate_type=self.jumprate_type,
                                   seed=self.seed, verbose=self.verbose)

        # Equilibration
        for sweep in xrange(self.equilibration_sweeps):
            if sweep % 1000 == 0:
                print >> self.output, "#Equilibration sweep {}/{}".format(sweep, self.equilibration_sweeps), "\r",
            if sweep % (self.skip_frames+1) == 0:
                if not self.shuffle:
                    Opos[:self.oxygennumber] = self.O_trajectory[sweep%self.O_trajectory.shape[0]]
                else:
                    Opos[:self.oxygennumber] = self.O_trajectory[np.random.randint(self.O_trajectory.shape[0])]
                self.extend_simulationbox(Opos, self.oxygennumber, self.h, self.box_multiplier)
                if self.po_angle:
                    Ppos[:self.phosphorusnumber] = self.P_trajectory[sweep%self.P_trajectory.shape[0]]
                    self.extend_simulationbox(Ppos, self.phosphorusnumber, self.h, self.box_multiplier)
                if sweep % neighbor_update == 0:
                    helper.determine_neighbors(Opos, self.cutoff_radius)
                if self.po_angle:
                    helper.calculate_transitions_POOangle(Opos, Ppos, self.P_neighbors, self.jumprate_params_fs, self.cutoff_radius, self.angle_threshold)
                else:
                    helper.calculate_transitions_new(Opos, self.jumprate_params_fs, self.cutoff_radius)
            helper.sweep_list(proton_lattice)

        if not self.xyz_output:
            self.print_observable_names()

        #~ print >> self.output, "#{:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10} {:>12}".format("Sweeps", "Time", "MSD_x", "MSD_y", "MSD_z", "Autocorr", "Jumps", "Sweeps/Sec", "Remaining Time/Min")
        #Run
        for sweep in xrange(0, self.sweeps):
            if sweep % (self.skip_frames+1) == 0:
                if not self.shuffle:
                    Opos[:self.oxygennumber] = self.O_trajectory[sweep%self.O_trajectory.shape[0]]
                else:
                    Opos[:self.oxygennumber] = self.O_trajectory[np.random.randint(self.O_trajectory.shape[0])]
                self.extend_simulationbox(Opos, self.oxygennumber, self.h, self.box_multiplier)
                if self.po_angle:
                    Ppos[:self.phosphorusnumber] = self.P_trajectory[sweep%self.P_trajectory.shape[0]]
                    self.extend_simulationbox(Ppos, self.phosphorusnumber, self.h, self.box_multiplier)
                if sweep % neighbor_update == 0:
                    # print >> self.output, "neighbor_update at sweep", sweep
                    helper.determine_neighbors(Opos, self.cutoff_radius)
                if self.po_angle:
                    helper.calculate_transitions_POOangle(Opos, Ppos, self.P_neighbors, self.jumprate_params_fs, self.cutoff_radius, self.angle_threshold)
                else:
                    helper.calculate_transitions_new(Opos, self.jumprate_params_fs, self.cutoff_radius)

            if sweep % self.reset_freq == 0:
                protonlattice_snapshot, proton_pos_snapshot = self.reset_observables(proton_pos_snapshot, proton_lattice, Opos, helper)
                # kMC_helper.calculate_transitionmatrix_nointra(transitionmatrix, Opos, self.jumprate_params_fs, self.MD_timestep_fs, self.pbc)
            if  sweep % self.print_freq == 0:
                #~ self.remove_com_movement_frame(Opos)
                if not self.nonortho:
                    self.calculate_displacement(proton_lattice, proton_pos_snapshot, proton_pos_new, Opos, displacement, self.pbc)
                else:
                    self.calculate_displacement_nonortho(proton_lattice, proton_pos_snapshot, proton_pos_new, Opos, displacement, self.h, self.h_inv)

                self.calculate_MSD(MSD, displacement)
                autocorrelation = self.calculate_autocorrelation(protonlattice_snapshot, proton_lattice)
                if not self.xyz_output:
                    self.print_observables(sweep, autocorrelation, helper, self.md_timestep_fs, start_time, MSD)
                else:
                    self.print_OsHs(Opos, proton_lattice, sweep, self.md_timestep_fs)
            helper.sweep_list(proton_lattice)

            # jumps = helper.sweep_gsl(proton_lattice, transitionmatrix)
            #~ jumps = helper.sweep_list_jumpmat(proton_lattice, jumpmatrix)

        #~ return jumpmatrix

def main(*args):
    parser=argparse.ArgumentParser(description="MDMC Test", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configfile", help="config file")
    args = parser.parse_args()
    md_mc = MDMC(configfile=args.configfile)
    # md_mc.print_settings()
    start_time = time.time()
    md_mc.ASEP()

    print "#Total time: {:.1f} minutes".format((time.time()-start_time)/60)

if __name__ == "__main__":
    main()
