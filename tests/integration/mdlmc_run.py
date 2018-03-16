import pathlib
from functools import partial
from itertools import tee

import daiquiri
import numpy as np

from mdlmc.IO.trajectory_parser import xyz_generator
from mdlmc.atoms.numpy_atom import NeighborTopology
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import Lattice, fastforward_to_next_jump, jumprate_generator


logger = daiquiri.getLogger(__name__)
daiquiri.getLogger("mdlmc.atoms").setLevel(daiquiri.logging.INFO)
daiquiri.setup(level=daiquiri.logging.DEBUG)


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) / c))


def test_mdlmc():
    pbc = [29.122, 25.354, 12.363]
    atombox = AtomBoxCubic(pbc)
    script_path = pathlib.Path(__file__).absolute().parent
    filename = "400Kbeginning.xyz"
    xyz_gen1, xyz_gen2 = tee(xyz_generator(script_path / filename, selection="O", repeat=True))
    topo = NeighborTopology(xyz_gen1, cutoff=3.0, buffer=1.0, atombox=atombox)

    lattice = Lattice(xyz_gen1, topo, 144, 96)

    jrf = partial(fermi, a=0.06, b=2.3, c=0.1)

    for frame, delta_frame, dt in fastforward_to_next_jump(jumprate_generator(jrf, lattice.lattice,
                                                                              topo),
                                                           dt=0.5):
        print(frame, delta_frame, dt)




if __name__ == "__main__":
    test_mdlmc()