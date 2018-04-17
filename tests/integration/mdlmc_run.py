import pathlib
from functools import partial
from itertools import tee

import daiquiri
import numpy as np

from mdlmc.IO.trajectory_parser import XYZTrajectory
from mdlmc.topo.topology import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import KMCLattice
from mdlmc.LMC.output import xyz_output, observables_output


logger = daiquiri.getLogger(__name__)
daiquiri.getLogger("mdlmc.topo").setLevel(daiquiri.logging.INFO)
daiquiri.setup(level=daiquiri.logging.INFO)


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) / c))


def test_mdlmc():
    format = "{: 10d} {:10.2f} {:16.8f} {:16.8f} {:16.8f} {:d}"
    pbc = [29.122, 25.354, 12.363]
    atombox = AtomBoxCubic(pbc)
    script_path = pathlib.Path(__file__).absolute().parent
    filename = "/home/kabbe/trajectories/400K.xyz"
    xyz_traj = XYZTrajectory(script_path / filename, selection="O", repeat=True, time_step=0.4)

    jrf = partial(fermi, a=0.06, b=2.3, c=0.1)
    kmc = KMCLattice(xyz_traj, atom_box=atombox, lattice_size=144, proton_number=96,
                     jumprate_function=jrf, donor_atoms="O")

    for frame_nr, time, msd, autocorr in observables_output(kmc, 1000, 10):
        print(format.format(frame_nr, time, *msd, autocorr))


if __name__ == "__main__":
    test_mdlmc()