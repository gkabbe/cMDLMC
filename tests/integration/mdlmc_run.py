# coding=utf-8

import pathlib

import daiquiri
import numpy as np
import pytest

from mdlmc.IO.trajectory_parser import XYZTrajectory, HDF5Trajectory
from mdlmc.topo.topology import (NeighborTopology, AngleTopology, HydroniumTopology,
                                 ReLUTransformation, DistanceInterpolator)
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.LMC.MDMC import KMCLattice
from mdlmc.LMC.jumprate_generators import Fermi, FermiAngle


logger = daiquiri.getLogger(__name__)
daiquiri.getLogger("mdlmc.topo").setLevel(daiquiri.logging.INFO)
daiquiri.setup(level=daiquiri.logging.INFO)


script_path = pathlib.Path(__file__).absolute().parent


np.random.seed(0)


extensions = [(XYZTrajectory, ".xyz"), (HDF5Trajectory, ".hdf5")]
@pytest.fixture(params=extensions)
def trajectory(request):
    Traj, ext = request.param
    return Traj(str(script_path / "trajectory") + ext, time_step=0.4)


@pytest.fixture
def atombox():
    pbc = [29.122, 25.354, 12.363]
    return AtomBoxCubic(pbc)


@pytest.fixture(params=[(Fermi, {"a": 0.06, "b": 2.3, "c": 0.1}),
                        (FermiAngle, {"a": 0.06, "b": 2.3, "c": 0.1, "theta": np.pi/2})])
def jumprate_function(request):
    JumpRate, jumprate_parameters = request.param
    return JumpRate(**jumprate_parameters)


@pytest.fixture(params=[(NeighborTopology, {}),
                        (AngleTopology, {"extra_atoms": "P", "group_size": 3}),
                        (HydroniumTopology, {})])
def topology(request, trajectory, atombox):
    Topo, extra_params = request.param
    return Topo(trajectory, atombox, donor_atoms="O", cutoff=3.0, buffer=2.0, **extra_params)


@pytest.mark.parametrize("output_type, output_params",
                         [("xyz_output", {}), ("observables_output", {"reset_frequency": 100,
                                                                      "print_frequency": 10})])
def test_mdlmc(topology, atombox, jumprate_function, output_type, output_params):
    if (type(topology), type(jumprate_function)) not in [(NeighborTopology, Fermi),
                                                         (AngleTopology, FermiAngle),
                                                         (HydroniumTopology, Fermi)]:
        pytest.skip(f"{topology.__class__} and {jumprate_function.__class__} not compatible.")

    logger.info("Topology: %s, Jumprate: %s", topology, jumprate_function)

    kmc = KMCLattice(topology, atom_box=atombox, lattice_size=144, proton_number=96,
                     jumprate_function=jumprate_function, donor_atoms="O", time_step=0.4)

    for frame in getattr(kmc, output_type)(**output_params):
        if output_type != "xyz_output":
            print(frame)


def test_hydronium(filename):
    trajectory = XYZTrajectory(filename, time_step=0.5)
    first_frame = next(iter(trajectory))["O"]
    oxygen_number = first_frame["O"].atom_number
    atombox = AtomBoxCubic([14.405] * 3)
    topology = HydroniumTopology(trajectory, atombox, donor_atoms="O", cutoff=4.0, buffer=2.0)
    transformation = ReLUTransformation(**rescale_parameters)
    kmc = KMCLattice(topology, atom_box=atombox, lattice_size=oxygen_number, proton_number=1,
                     jumprate_function=Fermi(a=0.06, b=2.3, c=0.1), donor_atoms="O", time_step=0.5)

    for frame in kmc.xyz_output():
        print(frame)
