import pathlib
from functools import partial
from itertools import tee

import daiquiri
import numpy as np

from mdlmc.IO.trajectory_parser import XYZTrajectory
from mdlmc.topo.topology import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import KMCLattice
from mdlmc.LMC.output import xyz_output


logger = daiquiri.getLogger(__name__)
daiquiri.getLogger("mdlmc.topo").setLevel(daiquiri.logging.INFO)
daiquiri.setup(level=daiquiri.logging.DEBUG)


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) / c))


def test_mdlmc():
    pbc = [29.122, 25.354, 12.363]
    atombox = AtomBoxCubic(pbc)
    script_path = pathlib.Path(__file__).absolute().parent
    filename = "400Kbeginning.xyz"
    xyz_traj = XYZTrajectory(script_path / filename, selection="O", repeat=True, time_step=0.4)

    jrf = partial(fermi, a=0.06, b=2.3, c=0.1)
    lattice = KMCLattice(xyz_traj, atom_box=atombox, lattice_size=144, proton_number=10,
                         jumprate_function=jrf, donor_atoms="O")

    for frame in xyz_output(kmc=lattice):
        print(frame)


if __name__ == "__main__":
    test_mdlmc()