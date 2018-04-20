import pathlib
from functools import partial
from itertools import tee

import daiquiri
import numpy as np

from mdlmc.IO.trajectory_parser import XYZTrajectory, HDF5Trajectory
from mdlmc.topo.topology import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import KMCLattice
from mdlmc.LMC.jumprate_generators import Fermi


logger = daiquiri.getLogger(__name__)
daiquiri.getLogger("mdlmc.topo").setLevel(daiquiri.logging.INFO)
daiquiri.setup(level=daiquiri.logging.INFO)


def test_mdlmc():
    format = "{: 10d} {:10.2f} {:16.8f} {:16.8f} {:16.8f} {:d}"
    pbc = [29.122, 25.354, 12.363]
    atombox = AtomBoxCubic(pbc)
    script_path = pathlib.Path(__file__).absolute().parent
    xyz_traj = XYZTrajectory(script_path / "400Kbeginning.xyz", selection="O", repeat=True, time_step=0.4)
    #hdf5traj = HDF5Trajectory(script_path / "400K.hdf5", selection="O", repeat=True, time_step=0.4, chunk_size=10000)
    topo = NeighborTopology(xyz_traj, atombox, donor_atoms="O", cutoff=3.0, buffer=2.0)

    jrf = Fermi(a=0.06, b=2.3, c=0.1)
    kmc = KMCLattice(topo, atom_box=atombox, lattice_size=144, proton_number=96,
                     jumprate_function=jrf, donor_atoms="O", time_step=0.4)

    #for frame_nr, time, msd, autocorr in kmc.continuous_output():
    #    print(format.format(frame_nr, time, *msd, autocorr))
    for frame in kmc.xyz_output():
        print(frame)


if __name__ == "__main__":
    test_mdlmc()
