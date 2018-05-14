from itertools import tee
import logging
from typing import Iterator


import numpy as np

from mdlmc.topo.topology import NeighborTopology
from ..misc.tools import remember_last_element
from ..LMC.output import CovalentAutocorrelation, MeanSquareDisplacement


logger = logging.getLogger(__name__)


def get_git_version():
    from mdlmc.version_hash import commit_hash, commit_date, commit_message
    print("# Hello. I am from commit {}".format(commit_hash))
    print("# Commit Date: {}".format(commit_date))
    print("# Commit Message: {}".format(commit_message))


class KMCLattice:
    def __init__(self, topology, *, lattice_size, atom_box, proton_number, jumprate_function,
                 donor_atoms, time_step, extra_atoms=None):
        """

        Parameters
        ----------
        trajectory
        lattice_size
        proton_number
        jumprate_function
        donor_atoms:
            name of donor / acceptor atoms
        extra_atoms:
            extra atoms used for the determination of the jump rate
        """

        self.topology = topology
        self._lattice = self._initialize_lattice(lattice_size, proton_number)
        # Check whether the topology object has the method "take_lattice_reference
        if hasattr(self.topology, "take_lattice_reference"):
            logger.debug("topology has method take_lattice_reference")
            self.topology.take_lattice_reference(self._lattice)
        self._atom_box = atom_box
        self._jumprate_function = jumprate_function
        self._donor_atoms = donor_atoms
        self._time_step = time_step
        self._extra_atoms = extra_atoms

    def _initialize_lattice(self, lattice_size, proton_number):
        lattice = np.zeros(lattice_size, dtype=np.int32)
        lattice[:proton_number] = range(1, proton_number + 1)
        np.random.shuffle(lattice)
        return lattice

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self.continuous_output()

    def continuous_output(self):
        current_frame_number = 0

        topo = self.topology
        topology_iterator, last_topo = remember_last_element(iter(self.topology))
        jumprate_iterator, last_jumprates = remember_last_element(
            jumprate_generator(self._jumprate_function, self.lattice, topology_iterator))
        sum_of_jumprates = (np.sum(jumpr) for _, _, jumpr in jumprate_iterator)
        kmc_routine = self.fastforward_to_next_jump(sum_of_jumprates,
                                                    self._time_step)

        for f, df, kmc_time in kmc_routine:
            current_time = kmc_time
            logger.debug("Next jump at time %.2f", current_time)
            logger.debug("df = %s; dt = %s", df, kmc_time)
            logger.debug("Go to frame %s", f)
            for frame in self.topology.get_cached_frames():
                yield current_frame_number, current_time, frame
                current_frame_number += 1

            proton_idx = self.move_proton(*last_jumprates())
            topo.update_time_of_last_jump(proton_idx, kmc_time)

    def move_proton(self, start, dest, jump_rates):
        """Given the hopping rates between the acceptor atoms, choose a connection randomly and
        move the proton."""

        cumsum = np.cumsum(jump_rates)
        random_draw = np.random.uniform(0, cumsum[-1])
        transition_idx = np.searchsorted(cumsum, random_draw)
        start_idx = start[transition_idx]
        destination_idx = dest[transition_idx]
        proton_idx = self._lattice[start_idx]
        logger.debug("Particle %s moves from %s to %s", proton_idx, start_idx, destination_idx)
        self._lattice[destination_idx] = proton_idx
        self._lattice[start_idx] = 0
        return proton_idx

    @staticmethod
    def fastforward_to_next_jump(jumprates, dt):
        """Implements Kinetic Monte Carlo with time-dependent rates.

        Parameters
        ----------
        jumprates : generator / iterator
            Unit: femtosecond^{-1}
            Proton jump rate from an oxygen site to any neighbor
        dt : float
            Trajectory time step

        Returns
        -------
        frame: int
            Frame at which the next event occurs
        delta_frame : int
            Difference between frame and the index at which the next event occurs
        kmc_time : float
            Time of the next event

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

    def xyz_output(self, particle_type: str = "H"):
        for f, t, frame in self:
            particle_positions = frame[self.donor_atoms][self.occupied_sites]
            particle_positions.atom_names = particle_type
            yield frame.append(particle_positions)

    def observables_output(self, reset_frequency: int, print_frequency: int):
        """

        Parameters
        ----------
        reset_frequency: int
        print_frequency: int

        Returns
        -------

        """
        kmc_iterator = iter(self)
        donor_sites = self.donor_atoms
        current_frame_number, current_time, frame = next(kmc_iterator)

        autocorr = CovalentAutocorrelation(self.lattice)
        msd = MeanSquareDisplacement(frame[donor_sites].atom_positions, self.lattice, self._atom_box)

        for current_frame_number, current_time, frame in kmc_iterator:
            if current_frame_number % reset_frequency == 0:
                autocorr.reset(self.lattice)
                msd.reset_displacement()

            msd.update_displacement(frame[donor_sites].atom_positions, self.lattice)

            if current_frame_number % print_frequency == 0:
                auto = autocorr.calculate(self.lattice)
                msd_result = msd.msd()
                yield current_frame_number, current_time, msd_result, auto


    @property
    def lattice(self):
        return self._lattice

    @property
    def donor_atoms(self):
        return self._donor_atoms

    @property
    def extra_atoms(self):
        return self._extra_atoms

    @property
    def occupied_sites(self):
        return np.where(self._lattice > 0)[0]


def jumprate_generator(jumprate_function, lattice, topology_iterator):

    for start, destination, *colvars in topology_iterator:
        omega = jumprate_function(*colvars)
        logger.debug("Omega shape: %s", omega.shape)
        # select only jumprates from donors which are occupied
        lattice_is_occupied = lattice > 0
        occupied_sites, = np.where(lattice_is_occupied)
        unoccupied_sites, = np.where(~lattice_is_occupied)
        occupied_mask = np.in1d(start, occupied_sites)
        unoccupied_mask = np.in1d(destination, unoccupied_sites)
        start_occupied_destination_free = occupied_mask & unoccupied_mask
        omega_allowed = omega[start_occupied_destination_free]
        start_allowed = start[start_occupied_destination_free]
        destination_allowed = destination[start_occupied_destination_free]
        yield start_allowed, destination_allowed, omega_allowed
