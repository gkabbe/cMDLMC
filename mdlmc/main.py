import argparse
import configparser
import logging.config
import pathlib

from .IO.trajectory_parser import XYZTrajectory, HDF5Trajectory
from .topo.topology import (NeighborTopology, AngleTopology, HydroniumTopology, ReLUTransformation,
                            InterpolatedTransformation, DistanceInterpolator)
from .cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic
from .LMC.jumprate_generators import Fermi, FermiAngle
from .LMC.MDMC import KMCLattice


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configfile", help="Text file containing the configuration for the cMD/LMC"
                                           "scheme")
    args = parser.parse_args()

    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(args.configfile, "r") as f:
        cp.read_file(f)

    # Check if logging config file exists in config file
    #if "Logging" in cp:
    #    logging_dict = cp["Logging"]
    #else:
    #    logfile_path = pathlib.Path(__file__).parents[1] / "logging.yaml"
    #    with open(logfile_path, "r") as f:
    #        logging_dict = yaml.load(f)
    #logging.config.dictConfig(logging_dict)
    #logger = logging.getLogger(__name__)

    #logger.debug(options)


    # setup trajectory
    trajectory_types = {"xyz": XYZTrajectory,
                        "hdf5": HDF5Trajectory}
    trajectory_options = dict(cp["Trajectory"])
    import ipdb; ipdb.set_trace()
    Trajectory = trajectory_types[trajectory_options.pop("type")]
    trajectory = Trajectory(**trajectory_options)

    # setup atom box
    atombox_options = options["AtomBox"]
    atombox_types = {"cubic": AtomBoxCubic,
                     "monoclinic": AtomBoxMonoclinic}
    AtomBox = atombox_types[atombox_options["type"]]
    atombox = AtomBox(atombox_options["periodic_boundaries"])

    # check if rescale options are specified
    distance_transformation_types = {"ReLUTransformation": ReLUTransformation,
                                     "InterpolatedTransformation": InterpolatedTransformation}
    if "Rescale" in options.keys():
        rescale_options = options["Rescale"]
        if "distance_transformation" in rescale_options:
            distance_transformation_options = rescale_options["distance_transformation"]
            DistanceTransformation = distance_transformation_types[distance_transformation_options["type"]]
            distance_transformation_parameters = distance_transformation_options["parameters"]
            distance_transformation = DistanceTransformation(**distance_transformation_parameters)
            relaxation_time = rescale_options["relaxation_time"]
            if relaxation_time == 0:
                distance_interpolator = None
            else:
                distance_interpolator = DistanceInterpolator(relaxation_time)

    # setup topology
    topology_options = options["Topology"]
    topology_types = {"HydroniumTopology": HydroniumTopology,
                      "NeighborTopology": NeighborTopology,
                      "AngleTopology": AngleTopology}
    Topology = topology_types[topology_options["type"]]
    topology_parameters = topology_options["parameters"]
    topology = Topology(trajectory, atombox, **topology_parameters)

    # setup jump rate
    jumprate_options = options["JumpRate"]
    jumprate_types = {"Fermi": Fermi, "FermiAngle": FermiAngle}
    jumprate_parameters = jumprate_options["parameters"]
    JumpRate = jumprate_types[jumprate_options["type"]]
    jumprate = JumpRate(**jumprate_parameters)

    # setup kmc
    kmc_options = options["KMC"]
    kmc_parameters = kmc_options["parameters"]
    kmc = KMCLattice(topology, atom_box=atombox, **kmc_parameters)

    # setup output
    output_options = options["Output"]
    output_type = output_options["type"]
    output_parameters = output_options["parameters"]

    for frame in getattr(kmc, output_type)(**output_parameters):
        print(frame)

