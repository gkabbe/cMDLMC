import pathlib
from functools import partial
from itertools import tee

import daiquiri
import numpy as np

from mdlmc.IO.trajectory_parser import XYZTrajectory
from mdlmc.topo.topology import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import KMCLattice


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

    for f, t, frame in lattice:
        print("Current frame:", f)
        print(t)
        print(frame.shape)
        proton_pos, = np.where(lattice.lattice)
        for pp in proton_pos:
            print(f"Proton {lattice.lattice[pp]} sits at position {pp}")


if __name__ == "__main__":
    test_mdlmc()