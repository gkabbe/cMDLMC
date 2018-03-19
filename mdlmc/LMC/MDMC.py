#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Iterator


import numpy as np
from ..IO.config_parser import print_confighelp, load_configfile, print_config_template, \
    check_cmdlmc_settings, print_settings
from ..IO.trajectory_parser import load_atoms
from ..atoms.numpy_atom import NeighborTopology
from ..cython_exts.LMC import LMCHelper
from ..cython_exts.LMC import PBCHelper
from mdlmc.cython_exts.LMC.LMCHelper import (ActivationEnergyFunction, FermiFunction,
                                             ExponentialFunction, FermiFunctionWater)


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
        self.displacement += self.atom_box.distance(self.proton_pos_snapshot, proton_pos_new)
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
                    self.atom_box.position_extended_box(oxy_ind, self.oxygen_trajectory[frame])

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
        self.averaged_results[(sweep % self.reset_freq) / self.print_freq, 2:] += \
            MSD[0], MSD[1], MSD[2], autocorrelation, helper.get_jumps()

    def print_xyz(self, Os, oxygen_lattice, sweep):
        proton_indices = np.where(oxygen_lattice > 0)[0]
        print(Os.shape[0] + self.proton_number, file=self.output)
        print("Time:", sweep * self.md_timestep, file=self.output)
        for i in range(Os.shape[0]):
            print("O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]), file=self.output)
        for index in proton_indices:
            print("H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]), file=self.output)


def initialize_oxygen_lattice(oxygen_number, proton_number):
    """Creates an integer array of length <oxygen_number> filled with randomly
    distributed numbers from 1 to <proton_number>

    Parameters
    ----------
    oxygen_number: int
                   The number of oxygen sites
    proton_number: int
                   The number of protons"""
    oxygen_lattice = np.zeros(oxygen_number, np.uint8)
    oxygen_lattice[:proton_number] = range(1, proton_number + 1)
    np.random.shuffle(oxygen_lattice)
    return oxygen_lattice


def prepare_lmc(settings):
    # First, print some info about the commit being used
    try:
        get_git_version()
    except ImportError:
        print("# No commit information found", file=sys.stderr)

    check_cmdlmc_settings(settings)
    verbose = settings.verbose

    if settings.o_neighbor:
        atoms_to_load = ("O", settings.o_neighbor)
    else:
        # Quick hack: use the oxyen trajectory as oxygen neighbor to initialize LMCRoutine,
        # but don't actually use it
        atoms_to_load = ("O", "O")
        print("# No oxygen neighbor specified, therefore the angle dependency will be switched off")
        settings.angle_dependency = False

    oxygen_trajectory, phosphorus_trajectory = load_atoms(settings.filename, *atoms_to_load,
                                                          auxiliary_file=settings.auxiliary_file,
                                                          clip=settings.clip_trajectory,
                                                          verbose=verbose, hdf5=settings.hdf5)
    if settings.seed is None:
        settings.seed = np.random.randint(2**32)
    np.random.seed(settings.seed)
    settings.oxygen_number = oxygen_trajectory.shape[1]
    settings.oxygen_number_extended = oxygen_trajectory.shape[1] * settings.box_multiplier[0] * \
        settings.box_multiplier[1] * settings.box_multiplier[2]
    # Multiply Arrhenius prefactor (unit 1/fs) with MD time step (unit fs), to get the
    # correct rates per time step
    if "A" in list(settings.jumprate_params_fs.keys()):
        settings.jumprate_params_fs["A"] *= settings.md_timestep_fs
    else:
        settings.jumprate_params_fs["a"] *= settings.md_timestep_fs
    # Check periodic boundaries and determine whether cell is orthorhombic/cubic
    #  or non-orthorhombic/monoclinic
    if len(settings.pbc) == 3:
        settings.nonortho = False
    else:
        settings.nonortho = True
    print_settings(settings)

    if settings.nonortho:
        if verbose:
            print("# Will use nonorthorhombic box")
        atom_box = PBCHelper.AtomBoxMonoclinic(settings.pbc, settings.box_multiplier)
    else:
        if verbose:
            print("# Will use orthorhombic box")
        atom_box = PBCHelper.AtomBoxCubic(settings.pbc, settings.box_multiplier)

    oxygen_lattice = initialize_oxygen_lattice(settings.oxygen_number_extended,
                                               settings.proton_number)
    msd_mode = "higher_msd" if settings.higher_msd else "standard_msd"
    if verbose:
        print("# Sweeps:", settings.sweeps, file=settings.output)

    # Jump rates determined via jumpstat
    if settings.jumprate_type == "MD_rates":
        a = settings.jumprate_params_fs["a"]
        b = settings.jumprate_params_fs["b"]
        c = settings.jumprate_params_fs["c"]
        jumprate_fct = FermiFunction(a, b, c)

    elif settings.jumprate_type == "MD_rates_Water":
        a = settings.jumprate_params_fs["a"]
        b = settings.jumprate_params_fs["b"]
        c = settings.jumprate_params_fs["c"]
        jumprate_fct = FermiFunctionWater(a, b, c)

    # Jump rates determined via energy surface scans
    elif settings.jumprate_type == "AE_rates":
        A = settings.jumprate_params_fs["A"]
        a = settings.jumprate_params_fs["a"]
        b = settings.jumprate_params_fs["b"]
        d0 = settings.jumprate_params_fs["d0"]
        T = settings.jumprate_params_fs["T"]
        jumprate_fct = ActivationEnergyFunction(A, a, b, d0, T)

    elif settings.jumprate_type == "Exponential_rates":
        a = settings.jumprate_params_fs["a"]
        b = settings.jumprate_params_fs["b"]
        jumprate_fct = ExponentialFunction(a, b)

    else:
        raise Exception("Jump rate type unknown. Please choose between "
                        "MD_rates, Exponential_rates and AE_rates")

    helper = LMCHelper.LMCRoutine(oxygen_trajectory, phosphorus_trajectory,
                                  atom_box=atom_box,
                                  jumprate_fct=jumprate_fct,
                                  cutoff_radius=settings.cutoff_radius,
                                  angle_threshold=settings.angle_threshold,
                                  neighbor_list=settings.neighbor_list,
                                  neighbor_search_radius=settings.neighbor_search_radius,
                                  verbose=settings.verbose, seed=settings.seed,
                                  angle_dependency=settings.angle_dependency)
    helper.store_jumprates(verbose=verbose)
    observable_manager = ObservableManager(helper, oxygen_trajectory, atom_box,
                                           oxygen_lattice, settings.proton_number,
                                           settings.md_timestep_fs, settings.sweeps,
                                           msd_mode=msd_mode,
                                           variance_per_proton=settings.variance_per_proton,
                                           output=settings.output)
    return oxygen_trajectory, oxygen_lattice, helper, observable_manager


def cmd_lmc_run(oxygen_trajectory, oxygen_lattice, helper, observable_manager, settings):
    """Main function. """

    verbose = settings.verbose

    # Equilibration
    for sweep in range(settings.equilibration_sweeps):
        if sweep % 1000 == 0:
            print("# Equilibration sweep {}/{}".format(sweep, settings.equilibration_sweeps),
                  end='\r', file=settings.output)
        if sweep % (settings.skip_frames + 1) == 0:
            if not settings.shuffle:
                frame = sweep
            else:
                frame = np.random.randint(oxygen_trajectory.shape[0])
        helper.sweep(sweep % oxygen_trajectory.shape[0], oxygen_lattice)
    if not settings.xyz_output:
        observable_manager.print_observable_names()

    # Run
    observable_manager.start_timer()
    for sweep in range(0, settings.sweeps):
        if sweep % (settings.skip_frames + 1) == 0:
            if not settings.shuffle:
                frame = sweep % oxygen_trajectory.shape[0]
            else:
                frame = np.random.randint(oxygen_trajectory.shape[0])

        if sweep % settings.reset_freq == 0:
            observable_manager.reset_observables(frame)

        if sweep % settings.print_freq == 0:
            if settings.xyz_output:
                observable_manager.print_xyz(oxygen_trajectory[frame], oxygen_lattice, sweep)
            else:
                observable_manager.calculate_displacement(frame)
                observable_manager.calculate_msd()
                observable_manager.calculate_auto_correlation()
                observable_manager.print_observables(sweep)

        if settings.jumpmatrix_filename is not None:
            helper.sweep_with_jumpmatrix(frame, oxygen_lattice)
        else:
            helper.sweep(frame, oxygen_lattice)

    if settings.jumpmatrix_filename is not None:
        np.savetxt(settings.jumpmatrix_filename, helper.jumpmatrix)


def main():
    parser = argparse.ArgumentParser(
        description="cMD/LMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_config_help = subparsers.add_parser("config_help", help="config file help")
    parser_config_file = subparsers.add_parser("config_file", help="Print config file template")
    parser_config_file.add_argument("--sorted", "-s", action="store_true",
                                    help="Sort config parameters lexicographically")
    parser_cmdlmc = subparsers.add_parser("cmdlmc", help="Load config file and start cMD/LMC run")
    parser_cmdlmc.add_argument("config_file", help="Config file")
    args = parser.parse_args()

    if args.subparser_name == "config_file":
        print_config_template(sort=args.sorted)
    elif args.subparser_name == "config_help":
        print_confighelp()
    else:
        settings = load_configfile(args.config_file, verbose=True)
        oxygen_trajectory, oxygen_lattice, helper, observable_manager = prepare_lmc(settings)
        cmd_lmc_run(oxygen_trajectory, oxygen_lattice, helper, observable_manager, settings)


class KMCLattice:
    def __init__(self, trajectory, topology, lattice_size, proton_number, jumprate_function,
                 donor_atoms, extra_atoms=None):
        """

        Parameters
        ----------
        trajectory
        topology
        lattice_size
        proton_number
        jumprate_function
        donor_atoms:
            name of donor / acceptor atoms
        extra_atoms:
            extra atoms used for the determination of the jump rate
        """

        self.trajectory = trajectory
        self.topology = topology
        self._initialize_lattice(lattice_size, proton_number)
        self.jumprate_function = jumprate_function
        self.donor_atoms = donor_atoms
        self.extra_atoms = extra_atoms

    @property
    def lattice(self):
        return self._lattice

    def _initialize_lattice(self, lattice_size, proton_number):
        self._lattice = np.zeros(lattice_size, dtype=np.int32)
        self._lattice[:proton_number] = range(1, proton_number + 1)
        np.random.shuffle(self._lattice)

    def __iter__(self) -> Iterator[np.ndarray]:
        current_frame = 0
        current_time  = 0

        jumprate_gen = self.jumprate_generator()
        kmc_routine = self.fastforward_to_next_jump(jumprate_gen, self.trajectory.time_step)

        for f, df, dt in kmc_routine:
            pass

    def fastforward_to_next_jump(self, jumprates, dt):
        """Implements Kinetic Monte Carlo with time-dependent rates.

        Parameters
        ----------
        jumprates : generator / iterator
            Unit: femtosecond^{-1}
            Proton jump rate from an oxygen site to any neighbor
        proton_position : int
            Index of oxygen at which the excess proton is residing.
        dt : float
            Trajectory time step
        frame : int
            Start frame
        time : float
            Start time

        Returns
        -------
        frame: int
            Frame at which the next event occurs
        delta_frame : int
            Difference between frame and the index at which the next event occurs
        delta_t : float
            Difference between current time and the time of the next event
        """

        sweep, kmc_time = 0, 0

        current_rate = next(jumprates)
        while True:
            time_selector = -np.log(1 - np.random.random())

            # Handle case where time selector is so small that the next frame is not reached
            t_trial = time_selector / current_rate
            if (kmc_time + t_trial) // dt == kmc_time // dt:
                kmc_time += t_trial
                delta_frame = 0
            else:
                delta_t, delta_frame = dt - kmc_time % dt, 1
                current_probsum = current_rate * delta_t
                next_rate = next(jumprates)
                next_probsum = current_probsum + next_rate * dt

                while next_probsum < time_selector:
                    delta_frame += 1
                    current_probsum = next_probsum
                    next_rate = next(jumprates)
                    next_probsum = current_probsum + next_rate * dt

                rest = time_selector - current_probsum
                delta_t += (delta_frame - 1) * dt + rest / next_rate
                kmc_time += delta_t
            sweep += delta_frame
            yield sweep, delta_frame, kmc_time

    def jumprate_generator(self):
        lattice = self.lattice
        topology = self.topology
        jumprate_function = self.jumprate_function

        for start, destination, distance in topology.topology_verlet_list_generator():
            omega = jumprate_function(distance)
            # select only jumprates from donors which are occupied
            occupied_sites, = np.where(lattice)
            yield omega[np.in1d(start, occupied_sites)].sum()


if __name__ == "__main__":
    main()
