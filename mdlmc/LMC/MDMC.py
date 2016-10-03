#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import argparse
from mdlmc.IO.xyz_parser import load_atoms
from mdlmc.IO.config_parser import print_confighelp, load_configfile, print_config_template
from mdlmc.cython_exts.LMC import LMCHelper
from mdlmc.cython_exts.LMC import PBCHelper


def get_git_version():
    from mdlmc.version_hash import commit_hash, commit_date, commit_message
    print("# Hello. I am from commit {}".format(commit_hash))
    print("# Commit Date: {}".format(commit_date))
    print("# Commit Message: {}".format(commit_message))


class ObservableManager:
    def __init__(self, lmc_helper, oxygen_trajectory, atom_box, oxygen_lattice, proton_number,
                 md_timestep, sweeps, *, msd_mode=None, variance_per_proton=False,
                 output=sys.stdout):
        self.lmc_helper = lmc_helper
        self.oxygen_trajectory = oxygen_trajectory
        self.atom_box = atom_box
        self.proton_number = proton_number
        self.displacement = np.zeros((self.proton_number, 3))
        self.proton_pos_snapshot = np.zeros((self.proton_number, 3))
        self.oxygen_lattice = oxygen_lattice
        self.oxygen_lattice_snapshot = np.array(oxygen_lattice)
        self.variance_per_proton = variance_per_proton
        self.md_timestep = md_timestep
        self.output = output
        self.sweeps = sweeps
        self.format_strings = ['{:10d}',    # Sweeps
                               '{:>10}',    # Time steps
                               '{:18.8f}',  # MSD x component
                               '{:18.8f}',  # MSD y component
                               '{:18.8f}',  # MSD z component
                               '{:}',       # MSD higher order
                               '{:8d}',     # OH bond autocorrelation
                               '{:10d}',    # Number of proton jumps
                               '{:10.2f}',  # Simulation speed
                               '{:}']       # Remaining time

        if msd_mode == "higher_msd":
            self.mean_square_displacement = np.zeros((4, 3))
            self.msd_variance = np.zeros((4, 3))
            self.calculate_msd = self.calculate_msd_higher_orders
        else:
            self.mean_square_displacement = np.zeros((1, 3))
            self.msd_variance = np.zeros(3)
            self.calculate_msd = self.calculate_msd_standard

    def calculate_displacement(self, frame):
        proton_pos_new = np.zeros(self.proton_pos_snapshot.shape)
        for oxygen_index, proton_index in enumerate(self.oxygen_lattice):
            if proton_index > 0:
                proton_pos_new[proton_index - 1] = \
                    self.atom_box.position_extended_box(oxygen_index,
                                                        self.oxygen_trajectory[frame])
        self.displacement += self.atom_box.distance(proton_pos_new,
                                                    self.proton_pos_snapshot)
        self.proton_pos_snapshot[:] = proton_pos_new

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
        self.autocorrelation = np.logical_and(self.oxygen_lattice == self.oxygen_lattice_snapshot,
                                              self.oxygen_lattice != 0).sum()

    def reset_observables(self, frame):
        for oxy_ind, prot_ind in enumerate(self.oxygen_lattice):
            if prot_ind > 0:
                self.proton_pos_snapshot[prot_ind - 1] = \
                    self.atom_box.position_extended_box(oxy_ind,
                                                        self.oxygen_trajectory[frame])

        self.oxygen_lattice_snapshot = np.copy(self.oxygen_lattice)

        self.displacement[:] = 0

    def print_observable_names(self):
        if self.variance_per_proton:
            print(
                "#     Sweeps       Time              MSD_x              MSD_y              MSD_z "
                "           MSD_x_var          MSD_y_var          MSD_z_var Autocorr      Jumps   "
                "Sweeps/Sec",
                file=self.output)
        else:
            print(
                "#     Sweeps       Time                 MSD_x              MSD_y              "
                "MSD_z Autocorr      Jumps   Sweeps/Sec",
                file=self.output)

    def print_observables(self, sweep):
        speed = float(sweep) / (time.time() - self.start_time)
        if sweep != 0:
            remaining_time_hours = int((self.sweeps - sweep) / speed / 3600)
            remaining_time_min = int((((self.sweeps - sweep) / speed) % 3600) / 60)
            remaining_time = "{:02d}:{:02d}".format(remaining_time_hours, remaining_time_min)
        else:
            remaining_time = "-01:-01"
        if self.mean_square_displacement.shape == (4, 3):
            msd2, msd3, msd4 = self.mean_square_displacement[1:]
            msd_higher = "{:18.8f} {:18.8f} {:18.8f}".format(msd2.sum(), msd3.sum(), msd4.sum())
        else:
            msd_higher = ""

        jump_counter = self.lmc_helper.jumps

        output = (sweep, sweep * self.md_timestep, *self.mean_square_displacement[0], msd_higher,
                  self.autocorrelation, jump_counter, speed, remaining_time)

        for i, (fmt_str, value) in enumerate(zip(self.format_strings, output)):
            print(fmt_str.format(value), end=" ", file=self.output)
        print(file=self.output)

    def start_timer(self):
        self.start_time = time.time()

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

    def print_xyz(self, Os, proton_lattice, sweep, timestep_fs):
        proton_indices = np.where(proton_lattice > 0)[0]
        print(Os.shape[0] + self.proton_number)
        print("Time:", sweep * timestep_fs)
        for i in range(Os.shape[0]):
            print("O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]))
        for index in proton_indices:
            print("H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]))


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

        self.oxygen_trajectory, self.phosphorus_trajectory = \
            load_atoms(self.filename, "O", self.o_neighbor, auxiliary_file=self.auxiliary_file,
                       clip=self.clip_trajectory, verbose=self.verbose)

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
        print("# I'm using the following settings:", file=self.output)
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
                print("# h = {} {} {}".format(*v[0]), file=self.output)
                print("#     {} {} {}".format(*v[1]), file=self.output)
                print("#     {} {} {}".format(*v[2]), file=self.output)
            elif k == "h_inv":
                print("# h_inv = {} {} {}".format(*v[0]), file=self.output)
                print("#         {} {} {}".format(*v[1]), file=self.output)
                print("#         {} {} {}".format(*v[2]), file=self.output)
            else:
                print("# {:20} {:>20}".format(k, str(v)), file=self.output)

    def initialize_oxygen_lattice(self, box_multiplier):
        """The oxygen lattice stores the occupation state of each oxygen.
        Protons are numbered from 1 to proton_number"""

        proton_lattice = np.zeros(
            self.oxygennumber * box_multiplier[0] * box_multiplier[1] * box_multiplier[2], np.uint8)
        proton_lattice[:self.proton_number] = range(1, self.proton_number + 1)
        np.random.shuffle(proton_lattice)

        return proton_lattice

    def cmd_lmc_run(self):
        """Main method. """

        # Check periodic boundaries and determine whether cell is orthorhombic/cubic
        #  or non-orthorhombic/monoclin
        if self.nonortho:
            atom_box = PBCHelper.AtomBoxMonoclinic(self.pbc, self.box_multiplier)
        else:
            atom_box = PBCHelper.AtomBoxCubic(self.pbc, self.box_multiplier)

        oxygen_lattice = self.initialize_oxygen_lattice(self.box_multiplier)

        msd_mode = "higher_msd" if self.higher_msd else "standard_msd"

        if self.verbose:
            print("# Sweeps:", self.sweeps, file=self.output)
        self.print_settings()

        helper = LMCHelper.LMCRoutine(self.oxygen_trajectory, self.phosphorus_trajectory,
                                      atom_box=atom_box,
                                      jumprate_parameter_dict=self.jumprate_params_fs,
                                      cutoff_radius=self.cutoff_radius,
                                      angle_threshold=self.angle_threshold,
                                      neighbor_search_radius=self.neighbor_search_radius,
                                      jumprate_type=self.jumprate_type, verbose=self.verbose)
        helper.store_jumprates(verbose=self.verbose)

        observable_manager = ObservableManager(helper, self.oxygen_trajectory, atom_box,
                                               oxygen_lattice, self.proton_number,
                                               self.md_timestep_fs, self.sweeps, msd_mode=msd_mode,
                                               variance_per_proton=self.variance_per_proton)

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
            helper.sweep(sweep % self.oxygen_trajectory.shape[0], oxygen_lattice)

        if not self.xyz_output:
            observable_manager.print_observable_names()

        # Run
        observable_manager.start_timer()
        for sweep in range(0, self.sweeps):
            if sweep % (self.skip_frames + 1) == 0:
                if not self.shuffle:
                    frame = sweep % self.oxygen_trajectory.shape[0]
                else:
                    frame = np.random.randint(self.oxygen_trajectory.shape[0])

            if sweep % self.reset_freq == 0:
                observable_manager.reset_observables(frame)
            if sweep % self.print_freq == 0:
                observable_manager.calculate_displacement(frame)
                observable_manager.calculate_msd()
                observable_manager.calculate_auto_correlation()
                observable_manager.print_observables(sweep)
            if self.jumpmatrix_filename is not None:
                helper.sweep_with_jumpmatrix(frame, oxygen_lattice)
            else:
                helper.sweep(frame, oxygen_lattice)

        if self.jumpmatrix_filename is not None:
            np.savetxt(self.jumpmatrix_filename, helper.jumpmatrix)


def start_lmc(args):
    md_mc = MDMC(configfile=args.config_file)
    md_mc.cmd_lmc_run()


def main(*args):
    parser = argparse.ArgumentParser(
        description="cMD/LMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    parser_config_help = subparsers.add_parser("config_help", help="config file help")
    parser_config_file = subparsers.add_parser("config_file", help="Print config file template")
    parser_config_load = subparsers.add_parser(
        "config_load", help="Load config file and start cMD/LMC run")
    parser_config_load.add_argument("config_file", help="Config file")
    parser_config_help.set_defaults(func=print_confighelp)
    parser_config_file.set_defaults(func=print_config_template)
    parser_config_load.set_defaults(func=start_lmc)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
