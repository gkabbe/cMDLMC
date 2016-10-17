#!/usr/bin/env python3
import argparse
import sys
import time

import numpy as np
from mdlmc.IO.config_parser import print_confighelp, load_configfile, print_config_template
from mdlmc.IO.xyz_parser import load_atoms
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
        self.averaged_results[(sweep % self.reset_freq) / self.print_freq, 2:] += MSD[0], MSD[1], \
                                                                                  MSD[2], \
                                                                                  autocorrelation, \
                                                                                  helper.get_jumps()

    def print_xyz(self, Os, oxygen_lattice, sweep, timestep_fs):
        proton_indices = np.where(oxygen_lattice > 0)[0]
        print(Os.shape[0] + self.proton_number, output=self.output)
        print("Time:", sweep * timestep_fs, output=self.output)
        for i in range(Os.shape[0]):
            print("O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]), output=self.output)
        for index in proton_indices:
            print("H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]), output=self.output)


def check_settings(settings):
    if settings.sweeps % settings.reset_freq != 0:
        raise ValueError("sweeps needs to be a multiple of reset_freq!")
    if settings.sweeps <= 0:
        raise ValueError("sweeps needs to be larger zero")


def print_settings(settings):
    print("# I'm using the following settings:", file=settings.output)
    for k, v in sorted(settings.__dict__.items()):
        if k == "h":
            print("# h = {} {} {}".format(*v[0]), file=settings.output)
            print("#     {} {} {}".format(*v[1]), file=settings.output)
            print("#     {} {} {}".format(*v[2]), file=settings.output)
        elif k == "h_inv":
            print("# h_inv = {} {} {}".format(*v[0]), file=settings.output)
            print("#         {} {} {}".format(*v[1]), file=settings.output)
            print("#         {} {} {}".format(*v[2]), file=settings.output)
        else:
            print("# {:20} {:>20}".format(k, str(v)), file=settings.output)


def initialize_oxygen_lattice(oxygen_number, proton_number):
    """The oxygen lattice stores the occupation state of each oxygen.
    Protons are numbered from 1 to proton_number"""
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

    check_settings(settings)
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
                                                          verbose=verbose)
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
    helper = LMCHelper.LMCRoutine(oxygen_trajectory, phosphorus_trajectory,
                                  atom_box=atom_box,
                                  jumprate_parameter_dict=settings.jumprate_params_fs,
                                  cutoff_radius=settings.cutoff_radius,
                                  angle_threshold=settings.angle_threshold,
                                  neighbor_search_radius=settings.neighbor_search_radius,
                                  jumprate_type=settings.jumprate_type, verbose=settings.verbose,
                                  seed=settings.seed, angle_dependency=settings.angle_dependency)
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
                observable_manager.print_xyz(oxygen_trajectory[frame], oxygen_lattice)
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


def kmc_state_to_xyz(oxygens, protons, oxygen_lattice):
    print(oxygens.shape[0] + protons.shape[0] + 1)
    print()
    for ox in oxygens:
        print("O", " ".join([3 * "{:14.8f}"]).format(*ox))
    for p in protons:
        print("H", " ".join([3 * "{:14.8f}"]).format(*p))
    oxygen_index = np.where(oxygen_lattice > 0)[0][0]
    print("S", " ".join([3 * "{:14.8f}"]).format(*oxygens[oxygen_index]))


def kmc_run(args):
    """Kinetic Monte Carlo run for a single excess charge"""
    def determine_probsum(probabilities, transition_indices, dt):
        total_transition_rate = np.array(probabilities)[transition_indices].sum()
        return total_transition_rate * dt

    oxygen_trajectory, oxygen_lattice, helper, observable_manager, settings = prepare_lmc(
        args.config_file)

    trajectory_length = oxygen_trajectory.shape[0]
    t, dt = 0, settings.md_timestep_fs
    frame, sweep, jumps = 0, 0, 0
    total_sweeps = settings.sweeps
    xyz_output = settings.xyz_output

    if xyz_output:
        proton_trajectory, = load_atoms(settings.filename, "H")

    while sweep < total_sweeps:
        proton_position = np.where(oxygen_lattice)[0][0]
        time_selector = -np.log(np.random.random())
        prob_sum = 0
        start, destination, probabilities = helper.return_transitions(frame)
        transition_indices, = np.where(np.array(start) == proton_position)
        prob_sum += determine_probsum(probabilities, transition_indices, dt)
        while prob_sum < time_selector:
            if xyz_output:
                kmc_state_to_xyz(oxygen_trajectory[frame], proton_trajectory[frame],
                                 oxygen_lattice)
            else:
                print("{:18d} {:18.2f} {:15.8f}"
                      " {:15.8f} {:15.8f} {:10d}".format(sweep, t,
                                                         *oxygen_trajectory[frame, proton_position],
                                                         jumps), flush=True)
            sweep, t = sweep + 1, t + dt
            frame = sweep % trajectory_length
            start, destination, probabilities = helper.return_transitions(frame)
            transition_indices, = np.where(np.array(start) == proton_position)
            prob_sum += determine_probsum(probabilities, transition_indices, dt)
        jumps += 1
        transition_probs = np.array(probabilities)[transition_indices]
        destination_indices = np.array(destination)[transition_indices]
        event_selector = np.random.random() * transition_probs.sum()
        transition_index = np.searchsorted(np.cumsum(transition_probs), event_selector)
        oxygen_lattice[proton_position] = 0
        proton_position = destination_indices[transition_index]
        oxygen_lattice[proton_position] = 1
        if xyz_output:
            kmc_state_to_xyz(oxygen_trajectory[frame], proton_trajectory[frame],
                             oxygen_lattice)
        else:
            print("{:18d} {:18.2f} {:15.8f}"
                  " {:15.8f} {:15.8f} {:10d}".format(sweep, t,
                                                     *oxygen_trajectory[frame, proton_position],
                                                     jumps), flush=True)


def main(*args):
    parser = argparse.ArgumentParser(
        description="cMD/LMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_config_help = subparsers.add_parser("config_help", help="config file help")
    parser_config_file = subparsers.add_parser("config_file", help="Print config file template")
    parser_config_file.add_argument("--sorted", "-s", action="store_true",
                                    help="Sort config parameters lexicographically")
    parser_cmdlmc = subparsers.add_parser("cmdlmc", help="Load config file and start cMD/LMC run")
    parser_cmdlmc.add_argument("config_file", help="Config file")
    parser_kmc = subparsers.add_parser("kmc", help="Load config file and start kmc run")
    parser_kmc.add_argument("config_file", help="Config file")
    parser_config_help.set_defaults(func=print_confighelp)
    parser_config_file.set_defaults(func=print_config_template)
    parser_cmdlmc.set_defaults(func=cmd_lmc_run)
    parser_kmc.set_defaults(func=kmc_run)
    args = parser.parse_args()

    if args.subparser_name == "config_file":
        print_config_template(args)
    elif args.subparser_name == "config_help":
        print_confighelp()
    else:
        settings = load_configfile(args.config_file, verbose=True)
        oxygen_trajectory, oxygen_lattice, helper, observable_manager = prepare_lmc(settings)
        args.func(oxygen_trajectory, oxygen_lattice, helper, observable_manager, settings)


if __name__ == "__main__":
    main()
