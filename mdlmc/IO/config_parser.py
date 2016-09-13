from collections import OrderedDict
from math import pi as PI
import re
import sys
from textwrap import wrap

import numpy as np


# Functions which parse the input from the config file
def get_jumprate_parameters(line):
    dict_string = re.findall("\{.*\}|dict\s*\(.*\)", line)[0]
    param_dict = eval(dict_string)
    return param_dict


def get_pbc(line):
    pbc = np.fromiter(map(float, line.split()[1:]), dtype=np.float)
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
        raise ValueError("Expected value is \"True\" or \"False\". Got {} instead".format(val))


def load_configfile(config_filename, verbose=False):
    parser_dict = CONFIG_DICT
    config_dict = dict()
    with open(config_filename, "r") as f:
        for line in f:
            if line[0] != "#":
                if len(line.split()) > 1:
                    keyword = line.split()[0]
                    if keyword in list(parser_dict.keys()):
                        config_dict[keyword.lower()] = parser_dict[keyword.lower()][
                            "parse_fct"](line)
                    else:
                        raise RuntimeError("Unknown keyword {}. Please remove it.".format(keyword))

    # Check for missing options, and look if they have a default argument
    for key, value in parser_dict.items():
        if key not in config_dict:
            if value["default"] == "no_default":
                raise RuntimeError("Missing value for {}".format(key))
            else:
                if verbose:
                    print("# Found no value for {} in config file".format(key))
                    print("# Will use default value: {}".format(value["default"]))
                config_dict[key] = value["default"]

    return config_dict


def print_confighelp(args):
    text_width = 80
    parser_dict = CONFIG_DICT
    for k, v in parser_dict.items():
        keylen = len(k)
        delim_len = (text_width - 2 - keylen) // 2
        print("{delim} {keyword} {delim}".format(keyword=k.upper(), delim=delim_len * "-"))
        print("")
        print("\n".join(wrap(v["help"], width=text_width)))
        print("")
        print("Default:", v["default"])
        print(text_width * "-")
        print("")
        print("")


CONFIG_DICT = OrderedDict([
    ("filename",
     {
         "parse_fct": parse_string,
         "default": None,
         "help": "Name of the coordinate file. If it has the ending .xyz, a new file with the "
                 "compressed coordinates and the ending .hd5 will be created."
                 "The hdf5 file can also be specified directly."
     }),
    ("auxiliary_file",
     {
         "parse_fct": parse_string,
         "default": None,
         "help": "Name of the coordinate file. If it has the ending .xyz, a new file with the "
                 "compressed coordinates and the ending .hd5 will be created."
                 "The hdf5 file can also be specified directly."
     }),
    ("output",
     {
         "parse_fct": string2file,
         "default": sys.stdout,
         "help": "Name of the output file"
     }),
    ("o_neighbor",
     {
         "parse_fct": parse_string,
         "default": "P",
         "help": "Name of the heavy atoms the oxygens are bonded to."
                 " Needed for the calculation of angle dependent jump rates."
     }),
    ("jumprate_type",
     {
         "parse_fct": parse_string,
         "default": "no_default",
         "help": "Choose between jump rates determined from static DFT activation energy "
                 "calculations (AE_rates) and jump rates determined from AIMD simulations ("
                 "MD_rates)."
     }),
    ("sweeps",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "Number of sweeps for the production run. A sweep is the number of single "
                 "proton jump attempts, after which (on average) each oxygen bond has been "
                 "selected once."
     }),
    ("equilibration_sweeps",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "Number of sweeps for the equilibration run."
     }),
    ("skip_frames",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "How many frames to skip when updating the topology from the MD trajectory. A "
                 "skip of zero frames means that all frames are read from the trajectory."
     }),
    ("print_freq",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "After how many sweeps should information about the system be printed?"
     }),
    ("reset_freq",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "After how many sweeps should quantities such as MSD and covalent bonding "
                 "autocorrelation function be reset? reset_freq should be a multiple of "
                 "print_freq, in order to ease the averaging of the final output."
     }),
    ("neighbor_search_radius",
     {
         "parse_fct": parse_float,
         "default": 15.0,
         "help": "All atoms whose distance is not larger than the neighbor_search_radius are "
                 "considered as neighbors."
     }),
    ("proton_number",
     {
         "parse_fct": parse_int,
         "default": "no_default",
         "help": "The number of acidic protons the KMC algorithm will use."
     }),
    ("clip_trajectory",
     {
         "parse_fct": parse_int,
         "default": None,
         "help": "Clip the number of frames used from the trajectory. If not specified, "
                 "the full trajectory will be used."
     }),
    ("seed",
     {
         "parse_fct": parse_int,
         "default": None,
         "help": "The seed for the random number generators. If none is specified, a random "
                 "one will be chosen."
     }),
    ("md_timestep_fs",
     {
         "parse_fct": parse_float,
         "default": None,
         "help": "Timestep of the used MD trajectory. Necessary to connect the KMC and MD time."
     }),
    ("angle_threshold",
     {
         "parse_fct": parse_float,
         "default": PI / 2,
         "help": "When using angle dependent jump rates, this option determines up to which "
                 "value of theta the jump rates will be set to zero. Theta is the angle "
                 "between the vector connecting an oxygen and its nearest heavy atom neighbor "
                 "(whose type can be defined via \"o_neighbor\") and the vector connecting the "
                 "two oxygens between which the jump rate is determined. Default is ninety "
                 "degrees."
     }),
    ("cutoff_radius",
     {
         "parse_fct": parse_float,
         "default": 4.0,
         "help": "Cutoff radius for the determination of jump rates. If two oxygens have a "
                 "larger distance, the jump rate will be set to zero."
     }),
    ("shuffle",
     {
         "parse_fct": parse_bool,
         "default": False,
         "help": "Whether to use shuffle mode, where frames from the trajectory are chosen "
                 "randomly."
     }),
    ("verbose",
     {
         "parse_fct": parse_bool,
         "default": False,
         "help": "Turn verbosity on or off."
     }),
    ("xyz_output",
     {
         "parse_fct": parse_bool,
         "default": False,
         "help": "Print xyz output."
     }),
    ("periodic_wrap",
     {
         "parse_fct": parse_bool,
         "default": True,
         "help": "If true, proton motion will be wrapped into the periodic box."
     }),
    ("jumpmatrix_filename",
     {
         "parse_fct": parse_string,
         "default": None,
         "help": "If a filename is given, the number of proton jumps between each oxygen pair "
                 "will be counted and saved."
     }),
    ("box_multiplier",
     {
         "parse_fct": lambda line: list(map(int, line.split()[1:])),
         "default": [1, 1, 1],
         "help": "Extend the LMC box along one or more dimensions."
     }),
    ("pbc",
     {
         "parse_fct": get_pbc,
         "default": "no_default",
         "help": "Specify the periodic boundary conditions of the MD trajectory. In the case of "
                 "orthogonal periodicity, three values describing the length of each box vector "
                 "are sufficient. In the case of non-orthogonal boxes, 9 entries are needed, "
                 "three for each unit cell vector."
     }),
    ("jumprate_params_fs",
     {
         "parse_fct": get_jumprate_parameters,
         "default": "no_default",
         "help": "Specify the parameters used for the calculation of the distance dependent "
                 "jump rate. If the jump rate type is \"MD_rates\", a dict containing values "
                 "for a, b, and c is expected (parameters for a fermi like step function ω(d) = "
                 "a / (1 + exp((x - b) / c)). "
                 ""
                 "If the jump rate type is \"AE_rates\", the expected "
                 "parameters are A, a, b, d0, and T. "
                 "a, b and d0 are fit parameters for the activation energy function "
                 "E(d) = a * (d - d0) / sqrt(b + 1 / (d - d0)**2), whereas A and T are parameters "
                 "of the Arrhenius equation , which converts the "
                 "activation energy to a jump rate: "
                 "ω(d) = A * exp(-E(d)/(k_B * T))"
     }),
    ("higher_msd",
     {
         "parse_fct": parse_bool,
         "default": False,
         "help": "Calculates higher MSDs."
     }),
    ("variance_per_proton",
     {
         "parse_fct": parse_bool,
         "default": False,
         "help": "If True, calculate variance over all proton trajectories. "
                 "Else, calculate variance over all time windows. In each time window, "
                 "the MSD has already been averaged over all proton trajectories."
     })
])
