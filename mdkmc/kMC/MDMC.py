#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import ipdb
import argparse
from textwrap import wrap
from mdkmc.IO import xyz_parser
from mdkmc.IO import config_parser
from mdkmc.cython_exts.kMC import kMC_helper
from mdkmc.cython_exts.atoms import numpyatom as npa


def load_configfile(configfilename, verbose=False):
    parser_dict = config_parser.CONFIG_DICT
    config_dict = dict()
    with open(configfilename, "r") as f:
        for line in f:
            if line[0] != "#":
                if len(line.split()) > 1:
                    keyword = line.split()[0]
                    if keyword in list(parser_dict.keys()):
                        config_dict[keyword.lower()] = parser_dict[keyword.lower()][
                            "parse_fct"](line)
                    else:
                        raise RuntimeError("Unknown keyword {}. Please remove it.".format(keyword))

    # Check for missing options, and look if they have a default argument
    for key, value in parser_dict.items():
        if key not in config_dict:
            if value["default"] == "no_default":
                raise RuntimeError("Missing value for {}".format(key))
            else:
                if verbose:
                    print("# Found no value for {} in config file".format(key))
                    print("# Will use default value: {}".format(value["default"]))
                config_dict[key] = value["default"]

    return config_dict


def print_confighelp(args):
    text_width = 80
    parser_dict = config_parser.CONFIG_DICT
    for k, v in parser_dict.items():
        keylen = len(k)
        delim_len = (text_width - 2 - keylen) // 2
        print("{delim} {keyword} {delim}".format(keyword=k.upper(), delim=delim_len * "-"))
        print("")
        print("\n".join(wrap(v["help"], width=text_width)))
        print("")
        print("Default:", v["default"])
        print(text_width * "-")
        print("")
        print("")


def get_gitversion():
    from mdkmc.version_hash import commit_hash, commit_date, commit_message
    print("# Hello. I am from commit {}".format(commit_hash))
    print("# Commit Date: {}".format(commit_date))
    print("# Commit Message: {}".format(commit_message))


def extend_simulationbox(oxygen_coordinates, oxygen_number, h, box_multiplier, nonortho=False):
    if True in [multiplier > 1 for multiplier in box_multiplier]:
        if nonortho:
            v1 = h[:, 0]
            v2 = h[:, 1]
            v3 = h[:, 2]
            Oview = oxygen_coordinates.view()
            Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], oxygen_number, 3)
            for x in range(box_multiplier[0]):
                for y in range(box_multiplier[1]):
                    for z in range(box_multiplier[2]):
                        if x + y + z != 0:
                            for i in range(oxygen_number):
                                Oview[x, y, z, i, :] = Oview[0, 0, 0, i] + x * v1 + y * v2 + z * v3
        else:
            ipdb.set_trace()
            Oview = oxygen_coordinates.view()
            Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], oxygen_number, 3)
            kMC_helper.extend_simulationbox(Oview, h,
                                            box_multiplier)


def calculate_displacement(proton_lattice, proton_lattice_snapshot,
                           oxygen_coordinates_new, displacement, pbc, wrap=True):
    proton_pos_new = np.zeros(proton_lattice_snapshot.shape)
    for oxygen_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index - 1] = oxygen_coordinates_new[oxygen_index]
    if wrap:
        kMC_helper.dist_numpy_all_inplace(displacement, proton_pos_new,
                                          proton_lattice_snapshot, pbc)
    else:
        displacement += kMC_helper.dist_numpy_all(proton_pos_new, proton_lattice_snapshot, pbc)
        proton_lattice_snapshot[:] = proton_pos_new  # [:] is important (inplace operation)


def calculate_displacement_nonortho(proton_lattice, proton_lattice_snapshot,
                                    Opos_new, displacement,
                                    h, h_inv):
    proton_pos_new = np.zeros(proton_lattice_snapshot.shape)
    for O_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index - 1] = Opos_new[O_index]
    kMC_helper.dist_numpy_all_nonortho(displacement, proton_pos_new, proton_lattice_snapshot, h, h_inv)


def calculate_autocorrelation(protonlattice_old, protonlattice_new):
    autocorrelation = 0
    for i in range(protonlattice_new.size):
        if protonlattice_old[i] == protonlattice_new[i] != 0:
            autocorrelation += 1
    return autocorrelation


def calculate_MSD(MSD, displacement):
    MSD *= 0
    for d in displacement:
        MSD += d * d
    MSD /= displacement.shape[0]


def calculate_MSD_var(MSD, displacement, msd_var):
    MSD *= 0
    msd_var *= 0
    for d in displacement:
        MSD += d * d
    MSD /= displacement.shape[0]
    for i in range(displacement.shape[1]):
        msd_var[i] = (displacement[:, i] * displacement[:, i]).var()
    return MSD, msd_var


def calculate_higher_MSD(displacement):
    MSD = np.zeros((3))
    msd2 = 0
    msd3 = 0
    msd4 = 0
    for d in displacement:
        MSD += d * d
        msd2 += (MSD[0] + MSD[1] + MSD[2]) ** 0.5
        msd3 += (MSD[0] + MSD[1] + MSD[2]) ** 1.5
        msd4 += (MSD[0] + MSD[1] + MSD[2]) ** 2
    MSD /= displacement.shape[0]
    msd2 /= displacement.shape[0]
    msd3 /= displacement.shape[0]
    msd4 /= displacement.shape[0]
    return MSD, msd2, msd3, msd4


def load_atoms(filename, auxiliary_file, clip, *atom_names, verbose=False):
    if filename:
        if auxiliary_file:
            if verbose:
                print("# Both xyz file and auxiliary npz file specified.")
                print("# Will try to load from auxiliary file", auxiliary_file)
            if not os.path.exists(auxiliary_file):
                if verbose:
                    print("# Specified auxiliary file does not exist.")
                    print("# Creating it now...")
                xyz_parser.save_trajectory_to_npz(filename, npz_fname=auxiliary_file,
                                                  remove_com_movement=True)
            return xyz_parser.load_trajectory_from_npz(auxiliary_file, *atom_names, clip=clip,
                                                       verbose=verbose)
        else:
            aux_fname = os.path.splitext(filename)[0] + ".npz"
            if verbose:
                print("# Only xyz file specified.")
                print("# Looking for auxiliary file", fname, "...")
            if os.path.exists(aux_fname):
                print("# Found it!")
                return xyz_parser.load_trajectory_from_npz(aux_fname, *atom_names, clip=clip,
                                                           verbose=verbose)
            else:
                if verbose:
                    print("# No auxiliary file found.")
                    print("# Will create it now...")
                    xyz_parser.save_trajectory_to_npz(filename, npz_fname=aux_fname,
                                                      remove_com_movement=True)
                    return xyz_parser.load_trajectory_from_npz(aux_fname, *atom_names, clip=clip,
                                                               verbose=verbose)
    else:
        print("# Found auxiliary file.")
        print("# Loading from there...")
        if auxiliary_file:
            return xyz_parser.load_trajectory_from_npz(auxiliary_file, *atom_names, clip=clip,
                                                       verbose=verbose)
        else:
            raise InputError("Please specify either filename or auxiliary_file")


class MDMC:
    def __init__(self, configfile):
        try:
            get_gitversion()
        except ImportError:
            print("# No commit information found", file=sys.stderr)
        file_kwargs = load_configfile(configfile, verbose=True)
        if ("verbose", True) in file_kwargs.items():
            print("# Config file specified. Loading settings from there.")

        # Save settings as object variable
        self.__dict__.update(file_kwargs)

        if self.po_angle:
            self.O_trajectory, self.P_trajectory = load_atoms(self.filename, self.auxiliary_file, self.clip_trajectory,
                                                              "O", "P",
                                                              verbose=self.verbose)
        else:
            self.O_trajectory = load_atoms(self.filename, self.auxiliary_file, self.clip_trajectory, "O",
                                           verbose=self.verbose)

        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(2 ** 32)
            np.random.seed(self.seed)

        self.oxygennumber = self.O_trajectory.shape[1]
        self.oxygennumber_extended = self.O_trajectory.shape[
                                         1] * self.box_multiplier[0] * self.box_multiplier[1] * self.box_multiplier[2]

        # Multiply Arrhenius prefactor (unit 1/fs) with MD time step (unit fs), to get the correct rates
        if "A" in list(self.jumprate_params_fs.keys()):
            self.jumprate_params_fs["A"] *= self.md_timestep_fs
        else:
            self.jumprate_params_fs["a"] *= self.md_timestep_fs

    def determine_phosphorus_oxygen_pairs(self, framenumber, atombox):
        P_neighbors = np.zeros(self.oxygennumber_extended, int)
        Os = atombox.get_extended_frame(atombox.oxygen_trajectory[framenumber])
        Ps = atombox.get_extended_frame(atombox.phosphorus_trajectory[framenumber])

        if self.nonortho:
            for i in range(Os.shape[0]):
                P_index = npa.nextNeighbor_nonortho(Os[i], Ps, atombox.h, atombox.h_inv)[0]
                P_neighbors[i] = P_index
        else:
            for i in range(Os.shape[0]):
                P_index = npa.nextNeighbor(Os[i], Ps, atombox.periodic_boundaries_extended)[0]
                P_neighbors[i] = P_index
        return P_neighbors

    def print_settings(self):
        print("# I'm using the following settings:")
        for k, v in self.__dict__.items():
            if "trajectory" in k:
                pass
            elif k == "angles":
                pass
            elif k == "P_neighbors":
                pass
            elif k == "default_dict":
                pass
            elif k == "h":
                print("# h = {} {} {}".format(*v[0]))
                print("#     {} {} {}".format(*v[1]))
                print("#     {} {} {}".format(*v[2]))
            elif k == "h_inv":
                print("# h_inv = {} {} {}".format(*v[0]))
                print("#         {} {} {}".format(*v[1]))
                print("#         {} {} {}".format(*v[2]))
            else:
                print("# {:20} {:>20}".format(k, str(v)))

    def init_proton_lattice(self, box_multiplier):
        proton_lattice = np.zeros(
            self.oxygennumber * box_multiplier[0] * box_multiplier[1] * box_multiplier[2], np.uint8)
        proton_lattice[:self.proton_number] = range(1, self.proton_number + 1)
        np.random.shuffle(proton_lattice)

        return proton_lattice

    def init_observables_protons_constant(self):
        displacement = np.zeros((self.proton_number, 3))
        MSD = np.zeros(3)
        proton_pos_snapshot = np.zeros((self.proton_number, 3))
        proton_pos_new = np.zeros((self.proton_number, 3))
        if not self.higher_msd:
            msd2, msd3, msd4 = None, None, None
        return displacement, MSD, msd2, msd3, msd4, proton_pos_snapshot, proton_pos_new

    def init_observables_protons_constant_var(self):
        displacement = np.zeros((self.proton_number, 3))
        msd_var = np.zeros(3)
        MSD = np.zeros(3)
        proton_pos_snapshot = np.zeros((self.proton_number, 3))
        proton_pos_new = np.zeros((self.proton_number, 3))
        if not self.higher_msd:
            msd2, msd3, msd4 = None, None, None
        return displacement, MSD, msd_var, msd2, msd3, msd4, proton_pos_snapshot, proton_pos_new

    def reset_observables(self, proton_pos_snapshot, protonlattice, displacement, Opos, helper):
        for i, site in enumerate(protonlattice):
            if site > 0:
                proton_pos_snapshot[site - 1] = Opos[i]

        protonlattice_snapshot = np.copy(protonlattice)

        helper.reset_jumpcounter()

        if not self.periodic_wrap:
            displacement[:] = 0

        return protonlattice_snapshot, proton_pos_snapshot, displacement

    def print_observable_names(self):
        if self.var_prot_single:
            print("# {:>10} {:>10} {:>18} {:>18} {:>18}   "
                  "{:>18} {:>18} {:>18} {:>8} {:>10} {:>12}".format("Sweeps", "Time", "MSD_x",
                                                                    "MSD_y", "MSD_z", "MSD_x_var",
                                                                    "MSD_y_var", "MSD_z_var",
                                                                    "Autocorr", "Jumps",
                                                                    "Sweeps/Sec",
                                                                    "Remaining Time/Hours:Min"),
                  file=self.output)
        else:
            print("# {:>10} {:>10}    {:>18} {:>18} {:>18} "
                  "{:>8} {:>10} {:>12}".format("Sweeps", "Time", "MSD_x", "MSD_y", "MSD_z",
                                               "Autocorr", "Jumps", "Sweeps/Sec",
                                               "Remaining Time/Hours:Min"), file=self.output)

    def print_observables(self, sweep, autocorrelation, helper, timestep_fs,
                          start_time, MSD, msd2=None, msd3=None, msd4=None):
        speed = float(sweep) / (time.time() - start_time)
        if sweep != 0:
            remaining_time_hours = int((self.sweeps - sweep) / speed / 3600)
            remaining_time_min = int((((self.sweeps - sweep) / speed) % 3600) / 60)
            remaining_time = "{:02d}:{:02d}".format(remaining_time_hours, remaining_time_min)
        else:
            remaining_time = "-01:-01"
        if (msd2, msd3, msd4) != (None, None, None):
            msd_higher = "{:18.8f} {:18.8f} {:18.8f}".format(msd2, msd3, msd4)
        else:
            msd_higher = ""

        print(" {:>10} {:>10}    "
              "{:18.8f} {:18.8f} {:18.8f}   "
              "{msd_higher:}  "
              "{:8d} {:10d} {:10.2f} {:10}".format(sweep, sweep * timestep_fs,
                                                   MSD[0], MSD[1], MSD[2],
                                                   autocorrelation, helper.get_jumps(), speed,
                                                   remaining_time, msd_higher=msd_higher),
              file=self.output)
        self.averaged_results[(sweep % self.reset_freq) / self.print_freq,
        2:] += MSD[0], MSD[1], MSD[2], autocorrelation, helper.get_jumps()

    def print_observables_var(self, sweep, autocorrelation, helper, timestep_fs,
                              start_time, MSD, msd_var, msd2=None, msd3=None, msd4=None):
        speed = float(sweep) / (time.time() - start_time)
        if sweep != 0:
            remaining_time_hours = int((self.sweeps - sweep) / speed / 3600)
            remaining_time_min = int((((self.sweeps - sweep) / speed) % 3600) / 60)
            remaining_time = "{:02d}:{:02d}".format(remaining_time_hours, remaining_time_min)
        else:
            remaining_time = "-01:-01"
        if (msd2, msd3, msd4) != (None, None, None):
            msd_higher = "{:18.8f} {:18.8f} {:18.8f}".format(msd2, msd3, msd4)
        else:
            msd_higher = ""
        print(" {:>10} {:>10}    "
              "{:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f}  "
              "{msd_higher:}  "
              "{:8d} {:10d} {:10.2f} {:10}".format(sweep, sweep * timestep_fs,
                                                   MSD[0], MSD[1], MSD[2],
                                                   msd_var[0], msd_var[1], msd_var[2],
                                                   autocorrelation, helper.get_jumps(), speed,
                                                   remaining_time, msd_higher=msd_higher),
              file=self.output)
        self.averaged_results[(sweep % self.reset_freq) / self.print_freq,
        2:] += MSD[0], MSD[1], MSD[2], autocorrelation, helper.get_jumps()

    def print_OsHs(self, Os, proton_lattice, sweep, timestep_fs):
        proton_indices = np.where(proton_lattice > 0)[0]
        print(Os.shape[0] + self.proton_number)
        print("Time:", sweep * timestep_fs)
        for i in range(Os.shape[0]):
            print("O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]))
        for index in proton_indices:
            print("H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]))

    def kmc_run(self):
        # Check periodic boundaries and determine whether cell is orthorhombic or non-orthorhombic
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

        if self.po_angle:
            self.P_neighbors = self.determine_phosphorus_oxygen_pairs(framenumber=0, atombox=atombox)

        if self.var_prot_single:
            displacement, MSD, msd_var, msd2, msd3, msd4, \
            proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant_var()
        else:
            displacement, MSD, msd2, msd3, msd4, \
            proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant()

        proton_lattice = self.init_proton_lattice(self.box_multiplier)

        if self.verbose:
            print("# Sweeps:", self.sweeps, file=self.output)
        self.print_settings()

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

        helper.determine_neighbors(0)
        helper.store_transitions_in_vector(verbose=self.verbose)

        self.averaged_results = np.zeros((self.reset_freq // self.print_freq, 7))

        # Equilibration
        for sweep in range(self.equilibration_sweeps):
            if sweep % 1000 == 0:
                print("# Equilibration sweep {}/{}".format(sweep, self.equilibration_sweeps),
                      end='\r', file=self.output)
            if sweep % (self.skip_frames + 1) == 0:
                if not self.shuffle:
                    frame = sweep
                else:
                    frame = np.random.randint(self.O_trajectory.shape[0])
            helper.sweep_from_vector(sweep % self.O_trajectory.shape[0], proton_lattice)

        if not self.xyz_output:
            self.print_observable_names()

        # Run
        start_time = time.time()
        for sweep in range(0, self.sweeps):
            if sweep % (self.skip_frames + 1) == 0:
                if not self.shuffle:
                    frame = sweep % self.O_trajectory.shape[0]
                else:
                    frame = np.random.randint(self.O_trajectory.shape[0])

            if sweep % self.reset_freq == 0:
                protonlattice_snapshot, proton_pos_snapshot, displacement = \
                    self.reset_observables(proton_pos_snapshot, proton_lattice, displacement,
                                           atombox.get_extended_frame(
                                               atombox.oxygen_trajectory[frame]), helper)
            if sweep % self.print_freq == 0:
                if not self.nonortho:
                    calculate_displacement(proton_lattice, proton_pos_snapshot,
                                           atombox.get_extended_frame(
                                               atombox.oxygen_trajectory[frame]),
                                           displacement, self.pbc, wrap=self.periodic_wrap)
                else:
                    calculate_displacement_nonortho(proton_lattice, proton_pos_snapshot,
                                                    atombox.get_extended_frame(
                                                        atombox.oxygen_trajectory[frame]),
                                                    displacement, atombox.h, atombox.h_inv)
                if self.higher_msd:
                    MSD, msd2, msd3, msd4 = calculate_higher_MSD(displacement)
                else:
                    if self.var_prot_single:
                        MSD, msd_var = calculate_MSD_var(MSD, displacement, msd_var)
                    else:
                        calculate_MSD(MSD, displacement)
                autocorrelation = calculate_autocorrelation(protonlattice_snapshot, proton_lattice)
                if not self.xyz_output:
                    if self.var_prot_single:
                        self.print_observables_var(sweep, autocorrelation, helper,
                                                   self.md_timestep_fs, start_time, MSD, msd_var,
                                                   msd2, msd3, msd4)
                    else:
                        self.print_observables(sweep, autocorrelation, helper, self.md_timestep_fs,
                                               start_time, MSD, msd2, msd3, msd4)
                else:
                    self.print_OsHs(atombox.oxygen_trajectory[
                                        frame], proton_lattice, frame, self.md_timestep_fs)
            # helper.sweep_list(proton_lattice)
            if self.jumpmatrix_filename is not None:
                helper.sweep_from_vector_jumpmatrix(frame, proton_lattice)
            else:
                helper.sweep_from_vector(frame, proton_lattice)

        if self.jumpmatrix_filename is not None:
            np.savetxt(self.jumpmatrix_filename, helper.jumpmatrix)

        self.averaged_results /= (self.sweeps / self.reset_freq)
        self.averaged_results[:, 0] = list(range(0, self.reset_freq, self.print_freq))
        self.averaged_results[:, 1] = self.averaged_results[:, 0] * self.md_timestep_fs
        print("# {}".format("-" * 98))
        print("# Averaged Results:")
        print("# {:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10}".format(
            "Sweeps", "Time", "MSD_x", "MSD_y", "MSD_z", "Autocorr", "Jumps"), file=self.output)
        for line in self.averaged_results:
            print("  {:>10} {:>10}    {:>18} {:>18} {:>18} {:>8} {:>10}".format(*line),
                  file=self.output)

        print("# Total time: {:.1f} minutes".format((time.time() - start_time) / 60),
              file=self.output)


def start_kmc(args):
    md_mc = MDMC(configfile=args.config_file)
    md_mc.kmc_run()


def main(*args):
    parser = argparse.ArgumentParser(
        description="MDMC Test", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    parser_config_help = subparsers.add_parser("config_help", help="config file help")
    parser_config_load = subparsers.add_parser(
        "config_load", help="Load config file and start KMC run")
    parser_config_load.add_argument("config_file", help="Config file")
    parser_config_help.set_defaults(func=print_confighelp)
    parser_config_load.set_defaults(func=start_kmc)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
