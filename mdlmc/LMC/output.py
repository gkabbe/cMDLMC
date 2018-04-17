import numpy as np

from ..LMC.MDMC import KMCLattice


class CovalentAutocorrelation:
    def __init__(self, lattice):
        self.reset(lattice)

    def reset(self, lattice):
        self.lattice = lattice.copy()

    def calculate(self, lattice):
        return np.sum((lattice == self.lattice) & (lattice != 0))


class MeanSquareDisplacement:
    def __init__(self, atom_positions, lattice, atombox):
        proton_number = np.sum(lattice > 0)
        self.snapshot = np.zeros((proton_number, 3))
        self.displacement = np.zeros_like(self.snapshot)
        self.snapshot = self.determine_proton_positions(atom_positions, lattice)
        self.atombox = atombox

    def determine_proton_positions(self, atom_positions, lattice):
        proton_positions = np.zeros_like(self.snapshot)
        proton_labels = lattice[lattice > 0]
        site_idx, = np.where(lattice)
        proton_positions[proton_labels - 1] = atom_positions[site_idx]
        return proton_positions

    def update_proton_positions(self, atom_positions, lattice):
        self.snapshot[:] = self.determine_proton_positions(atom_positions, lattice)

    def update_displacement(self, new_positions, lattice):
        """Update the current position of each proton while considering periodic boundaries.
        This assumes that the trajectory time step is small enough that no proton ever moves
        more than half of the periodic box length within one step."""

        new_proton_positions = self.determine_proton_positions(new_positions, lattice)
        displacement = self.atombox.distance(self.snapshot, new_proton_positions)
        self.displacement += displacement
        self.snapshot = new_proton_positions

    def reset_displacement(self):
        self.displacement[:] = 0

    def msd(self):
        return np.sum(self.displacement**2, axis=0) / self.displacement.shape[0]


def xyz_output(kmc: KMCLattice, particle_type: str = "H"):
    for f, t, frame in kmc:
        particle_positions = frame[kmc.occupied_sites]
        particle_positions.atom_names = particle_type
        yield frame.append(particle_positions)


def observables_output(kmc: KMCLattice, reset_frequency: int, print_frequency: int):
    """

    Parameters
    ----------
    kmc: KMCLattice
    reset_frequency: int
    print_frequency

    Returns
    -------

    """
    kmc_iterator = iter(kmc)
    donor_sites = kmc.donor_atoms
    _, _, first_frame = next(kmc_iterator)

    autocorr = CovalentAutocorrelation(kmc.lattice)
    msd = MeanSquareDisplacement(first_frame.select(donor_sites), kmc.lattice, kmc._atom_box)

    for current_frame_number, current_time, frame in kmc_iterator:
        if current_frame_number % reset_frequency == 0:
            autocorr.reset(kmc.lattice)
            msd.reset_displacement()

        msd.update_displacement(frame.select(donor_sites), kmc.lattice)

        if current_frame_number % print_frequency == 0:
            auto = autocorr.calculate(kmc.lattice)
            msd_result = msd.msd()
            yield current_frame_number, current_time, msd_result, auto

