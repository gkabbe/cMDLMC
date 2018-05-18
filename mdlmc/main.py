import argparse
import configparser
import inspect
import typing

import numpy as np

from .IO.trajectory_parser import XYZTrajectory, HDF5Trajectory
from .topo.topology import (NeighborTopology, AngleTopology, HydroniumTopology, ReLUTransformation,
                            InterpolatedTransformation, DistanceInterpolator)
from .cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic
from .LMC.jumprate_generators import Fermi, FermiAngle
from .LMC.MDMC import KMCLattice, XYZOutput, ObservablesOutput


def convert_to_match_signature(cls, keywords):
    keywords = dict(keywords)
    parameters = inspect.signature(cls).parameters
    for k in keywords:
        anno = parameters[k].annotation
        print(f"Convert {k} to {anno}")
        # Dirty hack to find out if the annotation is a union
        if anno.__class__.__name__ == "_Union":
            for type_ in anno.__args__:
                try:
                    keywords[k] = type_(keywords[k])
                except (ValueError, TypeError):
                    print(f"Could not convert {k} to {type_}")
                else:
                    print(f"Converted {k} to {type_}")
                    break
        elif keywords[k] == "EMPTY":
            raise ValueError(f"Keyword {k} is EMPTY. Please specify a value in the config file.")
        elif keywords[k] == "None":
            print(f"Convert {k} to None type")
            keywords[k] = None
        else:
            keywords[k] = parameters[k].annotation(keywords[k])
    return keywords


def eval_config(config_dict):
    for k in config_dict:
        try:
            config_dict[k] = eval(config_dict[k])
        except NameError:
            print("Skipping", k)


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
    trajectory_types = {"XYZTrajectory": XYZTrajectory,
                        "HDF5Trajectory": HDF5Trajectory}
    trajectory_options = dict(cp["Trajectory"])
    Trajectory = trajectory_types[trajectory_options.pop("type")]
    trajectory_options = convert_to_match_signature(Trajectory, trajectory_options)
    trajectory = Trajectory(**trajectory_options)

    # setup atom box
    atombox_options = dict(cp["AtomBox"])
    atombox_types = {"AtomBoxCubic": AtomBoxCubic,
                     "AtomBoxMonoclinic": AtomBoxMonoclinic}
    AtomBox = atombox_types[atombox_options.pop("type")]
    pbc = np.fromstring(atombox_options["periodic_boundaries"].strip("[]()"), dtype=float, sep=",")
    atombox = AtomBox(pbc)

    # check if distance interpolator is specified
    if "DistanceInterpolator" in cp:
        interpolator_options = cp["DistanceInterpolator"]
        interpolator_options = convert_to_match_signature(DistanceInterpolator, interpolator_options)
        relaxation_time = interpolator_options["relaxation_time"]
    else:
        relaxation_time = None

    # check if rescale options are specified
    distance_transformation_types = {"ReLUTransformation": ReLUTransformation,
                                     "InterpolatedTransformation": InterpolatedTransformation}
    if "DistanceTransformation" in cp:
        distance_transformation_options = dict(cp["DistanceTransformation"])
        transformation_type = distance_transformation_options.pop("type")
        DistanceTransformation = distance_transformation_types[transformation_type]
        distance_transformation_options = convert_to_match_signature(DistanceTransformation,
                                                                     distance_transformation_options)
        distance_transformation = DistanceTransformation(**distance_transformation_options)
        if relaxation_time:
            distance_interpolator = DistanceInterpolator(relaxation_time)
        else:
            distance_interpolator = None
    else:
        distance_transformation = None
        distance_interpolator = None


    # setup topology
    topology_options = dict(cp["NeighborTopology"])
    topology_types = {"HydroniumTopology": HydroniumTopology,
                      "NeighborTopology": NeighborTopology,
                      "AngleTopology": AngleTopology}
    topology_type = topology_options.pop("type")

    Topology = topology_types[topology_type]
    topology_options = convert_to_match_signature(Topology, topology_options)
    if topology_type == "HydroniumTopology":
        if not distance_transformation:
            raise NameError("Distance Transformation needs to be specified!")
        topology_options["distance_interpolator"] = distance_interpolator
        topology_options["distance_transformation_function"] = distance_transformation

    topology = Topology(trajectory, atombox, **topology_options)

    # setup jump rate
    jumprate_options = dict(cp["JumpRate"])
    jumprate_types = {"Fermi": Fermi, "FermiAngle": FermiAngle}
    JumpRate = jumprate_types[jumprate_options.pop("type")]
    jumprate_options = convert_to_match_signature(JumpRate, jumprate_options)
    #jumprate_options = convert_to_match_signature(JumpRate, jumprate_options)
    jumprate = JumpRate(**jumprate_options)

    # setup kmc
    kmc_options = dict(cp["KMCLattice"])
    kmc_options = convert_to_match_signature(KMCLattice, kmc_options)
    #kmc_options = convert_to_match_signature(KMCLattice, kmc_options)
    kmc = KMCLattice(topology, jumprate_function=jumprate, atom_box=atombox, **kmc_options)


    # setup output
    output_options = dict(cp["Output"])
    output_types = {"XYZOutput": XYZOutput,
                    "ObservablesOutput": ObservablesOutput}
    output_type = output_options.pop("type")
    Output = output_types[output_type]

    output = Output(kmc, **output_options)

    for x in output:
        print(x)

