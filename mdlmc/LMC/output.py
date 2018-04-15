import numpy as np

from ..LMC.MDMC import KMCLattice


def xyz_output(kmc: KMCLattice, particle_type: str = "H"):
    for f, t, frame in kmc:
        occupied = frame[kmc.occupied_sites]
        occupied.atom_names = particle_type
        yield frame.append(occupied)




