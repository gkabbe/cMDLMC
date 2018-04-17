import numpy as np

from ..LMC.MDMC import KMCLattice


class CovalentAutocorrelation:
    def __init__(self, lattice):
        self.update(lattice)

    def update(self, lattice):
        self.lattice = lattice

    def calculate(self, lattice):
        return np.sum((lattice == self.lattice) & (lattice != 0))


def xyz_output(kmc: KMCLattice, particle_type: str = "H"):
    for f, t, frame in kmc:
        occupied = frame[kmc.occupied_sites]
        occupied.atom_names = particle_type
        yield frame.append(occupied)




