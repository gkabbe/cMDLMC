from collections import namedtuple
from math import pi as PI
import re
import sys
import numpy as np


# Functions which parse the input from the config file
def get_jumprate_parameters(line):
    dict_string = re.findall("\{.*\}|dict\s*\(.*\)", line)[0]
    param_dict = eval(dict_string)
    return param_dict

def get_pbc(line):
    pbc = np.array(map(float, line.split()[1:]))
    if len(pbc) != 3 and len(pbc) != 9:
        raise ValueError("pbc length should be either 3 or 9")
    else:
        return pbc


def parse_int(line):
    try:
        return int(line.split()[1])
    except ValueError:
        return int(float(line.split()[1]))


def parse_float(line):
    return float(line.split()[1])


def parse_string(line):
    return line.split()[1]


def string2file(line, flag="w"):
    return open(line.split()[1], flag)


def parse_bool(line):
    val = line.split()[1]
    if val.upper() == "TRUE":
        return True
    elif val.upper() == "FALSE":
        return False
    else:
        print val
        raise(ValueError("Unknown value. Please use True or False"))


CONFIG_DICT = {
    "filename":
        {
            "parse_fct": parse_string,
            "default": "no_default",
            "help": "Name of the coordinate file"
        },
    "output":
        {
            "parse_fct": string2file,
            "default": sys.stdout,
            "help": "Name of the output file"
        },
    "o_neighbor":
        {
            "parse_fct": parse_string,
            "default": "P",
            "help": "Name of the heavy atoms the oxygens are bonded to."
                    " Needed for the calculation of angle dependent jump rates."
        },
    "jumprate_type":
        {
            "parse_fct": parse_string,
            "default": "no_default",
            "help": "Choose between jumprates determined from static DFT activation energy calculations (AE_rates) "
                    "and jumprates determined from AIMD simulations (MD_rates)."
        },
    "sweeps":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "Number of sweeps for the production run. A sweep is the number of single proton jump attempts, "
                    "after which (on average) each oxygen bond has been selected once."
        },
    "equilibration_sweeps":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "Number of sweeps for the equilibration run."
        },
    "skip_frames":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "How many frames to skip when updating the topology from the MD trajectory. A skip of zero frames "
                    "means that all frames are read from the trajectory."
        },
    "print_freq":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "After how many sweeps should information about the system be printed?"
        },
    "reset_freq":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "After how many sweeps should quantities such as MSD and covalent bonding autocorrelation function "
                    "be reset? reset_freq should be a multiple of print_freq, in order to ease the averaging of the "
                    "final output."
        },
    "neighbor_freq":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "After how many sweeps will the nearest neighbor connections be determined anew? Large numbers "
                    "will speed up the KMC run, but may distort the dynamics, as changes in the oxygen neighborhood "
                    "are detected less frequently."
        },
    "neighbor_search_radius":
        {
            "parse_fct": parse_float,
            "default": 15.0,
            "help": "All atoms whose distance is not larger than the neighbor_search_radius are considered as "
                    "neighbors."
        },
    "proton_number":
        {
            "parse_fct": parse_int,
            "default": "no_default",
            "help": "The number of acidic protons the KMC algorithm will use."
        },
    "clip_trajectory":
        {
            "parse_fct": parse_int,
            "default": None,
            "help": "Clip the number of frames used from the trajectory. If not specified, the full trajectory will be "
                    "used."
        },
    "seed":
        {
            "parse_fct": parse_int,
            "default": None,
            "help": "The seed for the random number generators. If none is specified, a random one will be chosen."
        },
    "md_timestep_fs":
        {
            "parse_fct": parse_float,
            "default": None,
            "help": "Timestep of the used MD trajectory. Necessary to connect the KMC and MD time."
        },
    "angle_threshold":
        {
            "parse_fct": parse_float,
            "default": PI/2,
            "help": "When using angle dependent jump rates, this option determines up to which value of theta the "
                    "jumprates will be set to zero. Theta is the angle between the vector connecting an oxygen and its "
                    "nearest heavy atom neighbor (whose type can be defined via \"o_neighbor\") and the vector "
                    "connecting the two oxygens between which the jumprate is determined. Default is ninety degrees."
        },
    "cutoff_radius":
        {
            "parse_fct": parse_float,
            "default": 4.0,
            "help": "Cutoff radius for the determination of jumprates. If two oxygens have a larger distance, the "
                    "jumprate will be set to zero."
        },
    "po_angle":
        {
            "parse_fct": parse_bool,
            "default": True,
            "help": "Whether to use angle_dependent jumprates or not."
        },
    "shuffle":
        {
            "parse_fct": parse_bool,
            "default": False,
            "help": "Whether to use shuffle mode, where frames from the trajectory are chosen randomly."
        },
    "verbose":
        {
            "parse_fct": parse_bool,
            "default": False,
            "help": "Turn verbosity on or off."
        },
    "xyz_output":
        {
            "parse_fct": parse_bool,
            "default": False,
            "help": "Print xyz output."
        },
    "periodic_wrap":
        {
            "parse_fct": parse_bool,
            "default": True,
            "help": "If true, proton motion will be wrapped into the periodic box."
        },
    "box_multiplier":
        {
            "parse_fct": lambda line: map(int, line.split()[1:]),
            "default": [1, 1, 1],
            "help": "Extend the KMC box along one or more dimensions."
        },
    "pbc":
        {
            "parse_fct": get_pbc,
            "default": "no_default",
            "help": "Set the periodic boundary conditions of the MD trajectory."
        },
    "jumprate_params_fs":
        {
            "parse_fct": get_jumprate_parameters,
            "default": "no_default",
            "help": "Specify the parameters used for the calculation of the distance dependent jumprate. If the"
                    "jumprate type is \"MD_rates\", a dict containing values for a, b, and c is expected (parameters"
                    "for a fermi like step function f(d) = a/(1+exp((x-b)/c) ). If the jumprate type is \"AE_rates\","
                    "the expected parameters are A, a, x0, xint and T. a, x0 and xint are fit parameters for the "
                    "function describing the activation energy over the oxygen distance, whereas A and T are parameters"
                    "of the Arrhenius equation, which converts the activation energy to a jumprate."
        }
}
