import argparse
import logging.config
import pathlib
import yaml

from .IO.trajectory_parser import XYZTrajectory, HDF5Trajectory
from .topo.topology import NeighborTopology, AngleTopology, HydroniumTopology
from .cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configfile", help="Text file containing the configuration for the cMD/LMC"
                                           "scheme")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        options = yaml.load(f)

    # Check if logging config file exists in config file
    if "Logging" in options:
        logging_dict = options["Logging"]
    else:
        logfile_path = pathlib.Path(__file__).parents[1] / "logging.yaml"
        with open(logfile_path, "r") as f:
            logging_dict = yaml.load(f)
    logging.config.dictConfig(logging_dict)
    logger = logging.getLogger(__name__)

    logger.debug(options)


    # setup trajectory
    trajectory_types = {"xyz": XYZTrajectory,
                        "hdf5": HDF5Trajectory}
    trajectory_options = options["Trajectory"]
    Trajectory = trajectory_types[trajectory_options["type"]]
    trajectory_parameters = trajectory_options["parameters"]
    trajectory = Trajectory(**trajectory_parameters)

    # setup atom box
    atombox_options = options["AtomBox"]
    atombox_types = {"cubic": AtomBoxCubic,
                     "monoclinic": AtomBoxMonoclinic}
    AtomBox = atombox_types[atombox_options[type]]
    atombox = AtomBoxCubic(atombox_options["periodic_boundaries"])

    # setup topology
    topology_options = options["Topology"]
    topology_types = {"HydroniumTopology": HydroniumTopology,
                      "NeighborTopology": NeighborTopology,
                      "AngleTopology": AngleTopology}
    Topology = topology_types[topology_options["type"]]
    topology_parameters = topology_options["parameters"]
    if topology_options["distance_transformation"]:
        distance_transformation_options = topology_options["distance_transformation"]
        transformation_types = {""}


    topology = Topology()