# coding=utf-8

from collections import OrderedDict
from math import pi as PI
import re
import sys
from textwrap import wrap

import numpy as np


# Functions which parse the input from the config file
def get_dictionary(line):
    dict_string = re.findall(r"\{.*\}|dict\s*\(.*\)", line)[0]
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


def parse_multifloat(line):
    return [float(x) for x in line.split()[1:]]


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


def load_configfile(config_filename, config_name="cMDLMC", verbose=False):
    class Settings:
        def __init__(self, dict_):
            self.__dict__.update(dict_)

        def __repr__(self):
            return self.__dict__.__repr__()

    parser_dict = CONFIG_DICT[config_name]
    config_dict = dict()
    with open(config_filename, "r") as f:
        if verbose:
            print("# Reading config file", config_filename)
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

    return Settings(config_dict)


def print_confighelp(config_name="cMDLMC"):
    text_width = 80
    parser_dict = CONFIG_DICT[config_name]
    horizontal, vertical = "-", "|"
    for k, v in parser_dict.items():
        print("{keyword:{sign}^{text_width}}".format(keyword=k.upper(), sign=horizontal,
                                                     text_width=text_width))
        print("{vertical}{: ^{}}{vertical}".format(" ", text_width-2, vertical=vertical))
        help_lines = wrap(v["help"], width=text_width - 4)
        for hl in help_lines:
            print("{vertical} {:<{}} {vertical}".format(hl, text_width - 4, vertical=vertical))
        print("{vertical}{: ^{}}{vertical}".format(" ", text_width-2, vertical=vertical))
        print("{vertical} {:<{}} {vertical}".format("Default: {}".format(v["default"]),
                                                    text_width - 4, vertical=vertical))
        print("{:{horizontal}^{text_width}}".format("", text_width=text_width,
                                                    horizontal=horizontal))
        print("*{: ^{}}*".format(" ", text_width-2))
        print("*{: ^{}}*".format(" ", text_width-2))


def print_config_template(config_name="cMDLMC", sort=False):
    parser_dict = CONFIG_DICT[config_name]
    items = parser_dict.items()
    if sort:
        items = sorted(items)
    for k, v in items:
        if v["default"] is None:
            print(k, "<MISSING VALUE>")
        else:
            print(k, v["default"])


def check_cmdlmc_settings(settings):
    if settings.sweeps % settings.reset_freq != 0:
        raise ValueError("sweeps needs to be a multiple of reset_freq!")
    if settings.sweeps <= 0:
        raise ValueError("sweeps needs to be larger zero")


def print_settings(settings):
    print("# I'm using the following settings:", file=settings.output)
    for k, v in sorted(settings.__dict__.items()):
        if k == "h":
            print("# h = {} {} {}".format(*v[0]), file=settings.output)
            print("#     {} {} {}".format(*v[1]), file=settings.output)
            print("#     {} {} {}".format(*v[2]), file=settings.output)
        elif k == "h_inv":
            print("# h_inv = {} {} {}".format(*v[0]), file=settings.output)
            print("#         {} {} {}".format(*v[1]), file=settings.output)
            print("#         {} {} {}".format(*v[2]), file=settings.output)
        else:
            print("# {:20} {:>20}".format(k, str(v)), file=settings.output)


CONFIG_DICT = {
    "cMDLMC": OrderedDict([
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
             "default": None,
             "help": "Name of the heavy atoms the oxygens are bonded to."
                     " Needed for the calculation of angle dependent jump rates."
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
             "default": 0,
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
        ("neighbor_list",
         {
             "parse_fct": parse_bool,
             "default": True,
             "help": "Use neighbor list to accelerate rate calculation. Should only be used in rather "
                     "rigid systems."
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
             "default": False,
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
        ("jumprate_type",
         {
             "parse_fct": parse_string,
             "default": "no_default",
             "help": "Choose between jump rates determined from static DFT activation energy "
                     "calculations (AE_rates), jump rates determined from AIMD simulations ("
                     "MD_rates), jump rates determined from AIMD simulations and special water distance criterion (MD_rates_Water) and Exponential_rates."
         }),
        ("jumprate_params_fs",
         {
             "parse_fct": get_dictionary,
             "default": "no_default",
             "help": "Specify the parameters used for the calculation of the distance dependent "
                     "jump rate. If jumprate_type is set to \"MD_rates\", a dict containing values "
                     "for a, b, and c is expected (parameters for a fermi like step function ω(d) = "
                     "a / (1 + exp((x - b) / c)). "
                     ""
                     "If jumprate_type is set to \"AE_rates\", the expected "
                     "parameters are A, a, b, d0, and T. "
                     "a, b and d0 are fit parameters for the activation energy function "
                     "E(d) = a * (d - d0) / sqrt(b + 1 / (d - d0)**2), whereas A and T are parameters "
                     "of the Arrhenius equation , which converts the "
                     "activation energy to a jump rate: "
                     "ω(d) = A * exp(-E(d) / (k_B * T))"
                     ""
                     "If jumprate_type is set to \"Exponential_rates\", the jump rate function is "
                     "defined as ω(d) = a * exp(b * x) with parameters a, b"
         }),
        ("higher_msd",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Calculates higher mean squared displacements."
         }),
        ("variance_per_proton",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "If True, calculate variance over all proton trajectories. "
                     "Else, calculate variance over all time windows. In each time window, "
                     "the MSD has already been averaged over all proton trajectories."
         }),
        ("angle_dependency",
         {
             "parse_fct": parse_bool,
             "default": True,
             "help": "If True, use angle cutoff (set jump rates to zero, if angle between oxygen 1 "
                     "neighbor, oxygen 1 and oxygen 2 is below angle_threshold."
         }),
        ("hdf5",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Save trajectory as hdf5 file. Recommended for large trajectories."
         })
    ]),
    "KMCWater": OrderedDict([
        ("filename",
         {
             "parse_fct": parse_string,
             "default": None,
             "help": "Name of the coordinate file. If it has the ending .xyz, a new file with the "
                     "compressed coordinates and the ending .hd5 will be created."
                     "The hdf5 file can also be specified directly."
         }),
        ("sweeps",
         {
             "parse_fct": parse_int,
             "default": None,
             "help": "Number of sweeps"
         }),
        ("print_frequency",
         {
             "parse_fct": parse_int,
             "default": 1,
             "help": "Print frequency. 1 means every frame, 2 every second and so on."
         }),
        ("chunk_size",
         {
             "parse_fct": parse_int,
             "default": 1000,
             "help": "Chunk size"
         }),
        ("relaxation_time",
         {
             "parse_fct": parse_int,
             "default": "no_default",
             "help": "Express the next-neighbor distances as a linear combination between the "
                     "relaxed and the unrelaxed distances. After a jump the distances are equal "
                     "to the unrelaxed distances and will decrease within <relaxation_time> to"
                     "the relaxed distances."
         }),
        ("waiting_time",
         {
             "parse_fct": parse_int,
             "default": 0,
             "help": "Waiting time after a proton jump."
         }),
        ("pbc",
         {
            "parse_fct": get_pbc,
             "default": None,
             "help": "Periodic box lengths"
         }),
        ("rescale_function",
         {
             "parse_fct": parse_string,
             "default": None,
             "help": "Specify the rescale function. Choices are 'linear' and 'ramp_function'."
                     "Accordingly, the expected rescale_parameters are a, b for linear and"
                     "a, b, d0 for ramp_function"
         }
         ),
        ("rescale_parameters",
         {
             "parse_fct": get_dictionary,
             "default": None,
             "help": "Parameters of the rescaling function for water, which transforms the O-O "
                     "distance"
                     "distribution of uncharged water molecules into the distance distribution "
                     "between"
                     "a hydronium ion and an uncharged water molecule."
         }),
        ("no_rescaling",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "If True, distances are not rescaled"
         }),
        ("xyz_output",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Output trajectory in XYZ format"
         }),
        ("jumprate_params_fs",
         {
             "parse_fct": get_dictionary,
             "default": None,
             "help": "Jump rate parameters"
         }),
        ("verbose",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Turn verbosity on or off."
         }),
        ("debug",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Enable debug information."
         }),
        ("overwrite_jumprates",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Overwrite jump rates in HDF5 file if True"
         }),
        ("md_timestep_fs",
         {
             "parse_fct": parse_float,
             "default": None,
             "help": "Time step of MD trajectory"
         }),
        ("output",
         {
             "parse_fct": string2file,
             "default": sys.stdout,
             "help": "Name of the output file"
         }),
        ("seed",
         {
             "parse_fct": parse_int,
             "default": None,
             "help": "Seed for random number generator"
         }),
        ("d_oh",
         {
             "parse_fct": parse_float,
             "default": None,
             "help": "Fix jump distance by substracting 2*d_oh along the jump vector during each "
                     "jump. If d_oh is None, the fix will not be applied."
         }),
        ("start_position",
         {
             "parse_fct": parse_int,
             "default": None,
             "help": "Set excess proton start position. If not specified, it will be set randomly"
         }),
        ("overwrite_oxygen_trajectory",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Overwrite oxygen trajectory in hdf5 file"
         }),
        ("mdconvert_trajectory",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "If trajectory is by mdconvert, it must be converted from nm to angstrom"
         }),
        ("keep_last_neighbor_rescaled",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "If True, the distance to the last donor stays rescaled after a jump. This way"
                     "the probability for a jump back is not diminished."
         }),
        ("check_from_old",
         {
             "parse_fct": parse_bool,
             "default": True,
             "help": "If True, and keep_last_neighbor_rescaled is True, check if the last oxygen"
                     " still has a connection to the current oxygen. If yes (and the current oxygen"
                     " has no connection back to the old one, replace another connection of the"
                     " current oxygen with the connection back to the old one"
         }),
        ("n_atoms",
         {
             "parse_fct": parse_int,
             "default": 3,
             "help": "Determines the number of closest neighbors that will be considered in the"
                     " KMC scheme."
         }),
        ("xyz_output",
         {
             "parse_fct": parse_bool,
             "default": False,
             "help": "Print xyz output."
         }),
        ("conversion_data",
         {
             "parse_fct": parse_string,
             "default": None,
             "help": "Filename with conversion data. First column is assumed to be distance, and"
                     " last column is assumed to be the conversion."
         })
    ])
}
