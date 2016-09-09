#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import ipdb
import argparse
from mdkmc.IO.xyz_parser import load_atoms
from mdkmc.IO.config_parser import print_confighelp, load_configfile
from mdkmc.cython_exts.LMC import LMCHelper
from mdkmc.cython_exts.atoms import numpyatom as npa


def get_git_version():
    from mdkmc.version_hash import commit_hash, commit_date, commit_message
    print("# Hello. I am from commit {}".format(commit_hash))
    print("# Commit Date: {}".format(commit_date))
    print("# Commit Message: {}".format(commit_message))


def calculate_displacement(proton_lattice, proton_lattice_snapshot,
                           oxygen_coordinates_new, displacement, pbc, wrap=True):
    proton_pos_new = np.zeros(proton_lattice_snapshot.shape)
    for oxygen_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index - 1] = oxygen_coordinates_new[oxygen_index]
    if wrap:
        LMCHelper.dist_numpy_all_inplace(displacement, proton_pos_new,
                                          proton_lattice_snapshot, pbc)
    else:
        displacement += LMCHelper.dist_numpy_all(proton_pos_new, proton_lattice_snapshot, pbc)
        proton_lattice_snapshot[:] = proton_pos_new  # [:] is important (inplace operation)


def calculate_displacement_nonortho(proton_lattice, proton_lattice_snapshot,
                                    Opos_new, displacement,
                                    h, h_inv):
    proton_pos_new = np.zeros(proton_lattice_snapshot.shape)
    for O_index, proton_index in enumerate(proton_lattice):
        if proton_index > 0:
            proton_pos_new[proton_index - 1] = Opos_new[O_index]
    LMCHelper.dist_numpy_all_nonortho(displacement, proton_pos_new, proton_lattice_snapshot, h,
                                       h_inv)


def calculate_auto_correlation(proton_lattice_old, proton_lattice_new):
    return np.logical_and(proton_lattice_old == proton_lattice_new, proton_lattice_new != 0).sum()


def calculate_mean_squared_displacement(mean_squared_displacement, displacement):
    mean_squared_displacement *= 0
    for d in displacement:
        mean_squared_displacement += d * d
    mean_squared_displacement /= displacement.shape[0]


def calculate_mean_squared_displacement_with_variance(mean_squared_displacement, displacement,
                                                      msd_variance):
    mean_squared_displacement *= 0
    msd_variance *= 0
    for d in displacement:
        mean_squared_displacement += d * d
    mean_squared_displacement /= displacement.shape[0]
    for i in range(displacement.shape[1]):
        msd_variance[i] = (displacement[:, i] * displacement[:, i]).var()
    return mean_squared_displacement, msd_variance


def calculate_higher_mean_squared_displacement(displacement):
    msd_1 = np.zeros((3))
    msd_2 = 0
    msd_3 = 0
    msd_4 = 0
    for d in displacement:
        msd_1 += d * d
        msd_2 += (msd_1[0] + msd_1[1] + msd_1[2])**0.5
        msd_3 += (msd_1[0] + msd_1[1] + msd_1[2])**1.5
        msd_4 += (msd_1[0] + msd_1[1] + msd_1[2])**2
    msd_1 /= displacement.shape[0]
    msd_2 /= displacement.shape[0]
    msd_3 /= displacement.shape[0]
    msd_4 /= displacement.shape[0]
    return msd_1, msd_2, msd_3, msd_4


class ObservableManager:
    def __init__(self, atom_box, proton_lattice, proton_number, *, msd_mode=None):
        self.proton_number = proton_number
        self.atom_box = atom_box
        self.displacement = np.zeros((self.proton_number, 3))
        self.proton_pos_snapshot = np.zeros((self.proton_number, 3))
        self.proton_lattice = proton_lattice
        self.proton_lattice_snapshot = np.array(proton_lattice)

        if msd_mode == "higher_msd":
            self.mean_square_displacement = np.zeros((4, 3))
            self.msd_variance = np.zeros((4, 3))
            self.calculate_msd = self.calculate_msd_higher_orders
        else:
            self.mean_square_displacement = np.zeros(3)
            self.msd_variance = np.zeros(3)
            self.calculate_msd = self.calculate_msd_standard

    def calculate_displacement(self):
        proton_pos_new = np.zeros(self.proton_lattice_snapshot.shape)
        for oxygen_index, proton_index in enumerate(self.proton_lattice):
            if proton_index > 0:
                proton_pos_new[proton_index - 1] = self.atom_box.oxygen_coordinates_new[
                    oxygen_index]
        self.displacement += LMCHelper.dist_numpy_all(proton_pos_new, self.proton_lattice_snapshot,
                                                       pbc)
        self.proton_lattice_snapshot[:] = self.proton_lattice

    def calculate_msd_standard(self):
        self.mean_square_displacement[:] = (self.displacement**2).sum(axis=0) / \
                                           self.displacement.shape[0]
        return self.mean_square_displacement

    def calculate_msd_higher_orders(self):
        self.mean_square_displacement[:] = 0
        self.mean_square_displacement[0] = (self.displacement**2).sum(axis=0)
        self.mean_square_displacement[1] = self.mean_square_displacement[0].sum()**0.5
        self.mean_square_displacement[2] = self.mean_square_displacement[1].sum()**3
        self.mean_square_displacement[3] = self.mean_square_displacement[1].sum()**4
        self.mean_square_displacement /= self.displacement.shape[0]

        return self.mean_square_displacement

    def calculate_auto_correlation(self):
        return np.logical_and(self.proton_lattice == self.proton_lattice_snapshot,
                              self.proton_lattice != 0).sum()

    def return_observables(self, *observables):
        self.displacement[:] = 0


class MDMC:
    def __init__(self, configfile):
        try:
            get_git_version()
        except ImportError:
            print("# No commit information found", file=sys.stderr)
        file_kwargs = load_configfile(configfile, verbose=True)
        if ("verbose", True) in file_kwargs.items():
            print("# Config file specified. Loading settings from there.")

        # Save settings as object variable
        self.__dict__.update(file_kwargs)

        self.oxygen_trajectory, self.phosphorus_trajectory = load_atoms(self.filename,
                                                                        self.auxiliary_file,
                                                                        "O", "P",
                                                                        clip=self.clip_trajectory,
                                                                        verbose=self.verbose)

        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(2**32)
            np.random.seed(self.seed)

        self.oxygennumber = self.oxygen_trajectory.shape[1]
        self.oxygennumber_extended = self.oxygen_trajectory.shape[1] * self.box_multiplier[0] * \
                                     self.box_multiplier[1] * \
                                     self.box_multiplier[2]

        # Multiply Arrhenius prefactor (unit 1/fs) with MD time step (unit fs), to get the
        # correct rates
        if "A" in list(self.jumprate_params_fs.keys()):
            self.jumprate_params_fs["A"] *= self.md_timestep_fs
        else:
            self.jumprate_params_fs["a"] *= self.md_timestep_fs

        if len(self.pbc) == 3:
            self.nonortho = False
        else:
            self.nonortho = True

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

    def initialize_proton_lattice(self, box_multiplier):
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

    def reset_observables(self, proton_position_snapshot, proton_lattice, displacement,
                          oxygen_positions, helper):
        for i, site in enumerate(proton_lattice):
            if site > 0:
                proton_position_snapshot[site - 1] = oxygen_positions[i]

        protonlattice_snapshot = np.copy(proton_lattice)

        helper.reset_jumpcounter()

        if not self.periodic_wrap:
            displacement[:] = 0

        return protonlattice_snapshot, proton_position_snapshot, displacement

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
        self.averaged_results[(sweep % self.reset_freq) // self.print_freq, 2:] += MSD[0], MSD[1], \
                                                                                   MSD[2], \
                                                                                   autocorrelation, \
                                                                                   helper.get_jumps()

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
        self.averaged_results[(sweep % self.reset_freq) / self.print_freq, 2:] += MSD[0], MSD[1], \
                                                                                  MSD[2], \
                                                                                  autocorrelation, \
                                                                                  helper.get_jumps()

    def print_OsHs(self, Os, proton_lattice, sweep, timestep_fs):
        proton_indices = np.where(proton_lattice > 0)[0]
        print(Os.shape[0] + self.proton_number)
        print("Time:", sweep * timestep_fs)
        for i in range(Os.shape[0]):
            print("O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]))
        for index in proton_indices:
            print("H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]))

    def cmd_lmc_run(self):
        """Main method. """

        # Check periodic boundaries and determine whether cell is orthorhombic/cubic
        #  or non-orthorhombic/monoclin
        if self.nonortho:
            atombox = LMCHelper.AtomBoxMonoclin(self.oxygen_trajectory,
                                                  self.phosphorus_trajectory, self.pbc,
                                                  self.box_multiplier)
        else:
            atombox = LMCHelper.AtomBoxCubic(self.oxygen_trajectory, self.phosphorus_trajectory,
                                               self.pbc, self.box_multiplier)

        if self.var_prot_single:
            displacement, MSD, msd_var, msd2, msd3, msd4, \
            proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant_var()
        else:
            displacement, MSD, msd2, msd3, msd4, \
            proton_pos_snapshot, proton_pos_new = self.init_observables_protons_constant()

        proton_lattice = self.initialize_proton_lattice(self.box_multiplier)

        msd_mode = "higher_msd" if self.higher_msd else "standard_msd"
        observable_manager = ObservableManager(atombox, proton_lattice, self.proton_number,
                                               msd_mode=msd_mode)

        if self.verbose:
            print("# Sweeps:", self.sweeps, file=self.output)
        self.print_settings()

        start_time = time.time()

        helper = LMCHelper.LMCRoutine(atombox=atombox,
                                       jumprate_parameter_dict=self.jumprate_params_fs,
                                       cutoff_radius=self.cutoff_radius,
                                       angle_threshold=self.angle_threshold,
                                       neighbor_search_radius=self.neighbor_search_radius,
                                       jumprate_type=self.jumprate_type, verbose=self.verbose)

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
                    frame = np.random.randint(self.oxygen_trajectory.shape[0])
            helper.sweep_from_vector(sweep % self.oxygen_trajectory.shape[0], proton_lattice)

        if not self.xyz_output:
            self.print_observable_names()

        # Run
        start_time = time.time()
        for sweep in range(0, self.sweeps):
            if sweep % (self.skip_frames + 1) == 0:
                if not self.shuffle:
                    frame = sweep % self.oxygen_trajectory.shape[0]
                else:
                    frame = np.random.randint(self.oxygen_trajectory.shape[0])

            if sweep % self.reset_freq == 0:
                proton_lattice_snapshot, proton_pos_snapshot, displacement = \
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
                    MSD, msd2, msd3, msd4 = calculate_higher_mean_squared_displacement(displacement)
                else:
                    if self.var_prot_single:
                        MSD, msd_var = calculate_mean_squared_displacement_with_variance(MSD,
                                                                                         displacement,
                                                                                         msd_var)
                    else:
                        calculate_mean_squared_displacement(MSD, displacement)
                auto_correlation = calculate_auto_correlation(proton_lattice_snapshot,
                                                              proton_lattice)
                if not self.xyz_output:
                    if self.var_prot_single:
                        self.print_observables_var(sweep, auto_correlation, helper,
                                                   self.md_timestep_fs, start_time, MSD, msd_var,
                                                   msd2, msd3, msd4)
                    else:
                        self.print_observables(sweep, auto_correlation, helper, self.md_timestep_fs,
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


def start_lmc(args):
    md_mc = MDMC(configfile=args.config_file)
    md_mc.cmd_lmc_run()


def main(*args):
    parser = argparse.ArgumentParser(
        description="cMD/LMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    parser_config_help = subparsers.add_parser("config_help", help="config file help")
    parser_config_load = subparsers.add_parser(
        "config_load", help="Load config file and start cMD/LMC run")
    parser_config_load.add_argument("config_file", help="Config file")
    parser_config_help.set_defaults(func=print_confighelp)
    parser_config_load.set_defaults(func=start_lmc)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
