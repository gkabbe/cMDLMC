import argparse
import sys
import logging

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d

from mdlmc.IO import xyz_parser
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic
from mdlmc.misc.tools import argparse_compatible
from mdlmc.KMC.excess_kmc import rescale_interpolation_function


logger = logging.getLogger(__name__)


def reduce_pbc(atom_pos, origin, pbc):
    mask = atom_pos - origin < -pbc / 2
    while mask.any():
        atom_pos[mask] += pbc[mask]
        mask = atom_pos - origin < -pbc / 2
    mask = atom_pos - origin > pbc / 2
    while mask.any():
        atom_pos[mask] -= pbc[mask]
        mask = atom_pos - origin > pbc / 2


def rescale_distance(atom_pos, origin, new_distance):
    connection_vec = atom_pos - origin
    new_pos = origin + connection_vec / np.linalg.norm(connection_vec) * new_distance
    return new_pos


def excess_charge_collective_variable(oxygen_pos, proton_pos):
    return proton_pos.sum(axis=0) - 2 * oxygen_pos.sum(axis=0)


def group_h2os(oxygens, protons, atombox):
    dist_matrix = atombox.length_all_to_all(oxygens, protons)
    closest_protons = np.argsort(dist_matrix, axis=-1)[:, :2]
    return closest_protons


def print_h2o(oxygen, proton1, proton2, pbc):
    print("O", *oxygen)
    reduce_pbc(proton1, oxygen, pbc)
    print("H", *proton1)
    reduce_pbc(proton2, oxygen, pbc)
    print("H", *proton2)


class HydroniumHelper:
    def __init__(self):
        self.last_hydronium_index = -1
        self.jumps = -1

    def determine_hydronium_index(self, oxygen_pos, proton_pos, atombox):
        closest_oxygen_indices = np.zeros(proton_pos.shape[0], dtype=int)
        for proton_index, proton in enumerate(proton_pos):
            oxygen_index, _ = atombox.next_neighbor(proton, oxygen_pos)
            closest_oxygen_indices[proton_index] = oxygen_index
        for i in range(oxygen_pos.shape[0]):
            if (closest_oxygen_indices == i).sum() == 3:
                if i != self.last_hydronium_index:
                    self.last_hydronium_index = i
                    self.jumps += 1
                return i
        else:
            raise RuntimeError("Could not determine excess charge index.")


@argparse_compatible
def track_collective_variable(trajectory, pbc, *, visualize=False):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_atoms(trajectory, "O", "H")
    hydronium_helper = HydroniumHelper()
    excess_charge_start_index = hydronium_helper.determine_hydronium_index(oxygens[0], protons[0],
                                                                           atombox)
    excess_charge_start_position = oxygens[0, excess_charge_start_index]
    excess_charge_colvar_0 = excess_charge_collective_variable(oxygens[0], protons[0])
    if visualize:
        atoms = xyz_parser.load_atoms(trajectory)

    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        # Like in the PVPA paper, create a collective variable that tracks the excess charge motion
        excess_charge_colvar = excess_charge_collective_variable(oxygen_frame, proton_frame) \
                               - excess_charge_colvar_0 + excess_charge_start_position
        if visualize:
            print(len(atoms[i]) + 1)
            print()
            for atom in atoms[i]:
                print(atom["name"], " ".join(map(str, atom["pos"])))
            print("S", " ".join(map(str, excess_charge_colvar)), flush=True)
        else:
            print("1")
            print()
            print("S", " ".join(map(str, excess_charge_colvar)), flush=True)


@argparse_compatible
def track_hydronium_ion(trajectory, pbc, *, visualize=False):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_atoms(trajectory, "O", "H")
    if visualize:
        atoms = xyz_parser.load_atoms(trajectory)
    hydronium_helper = HydroniumHelper()
    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame, proton_frame,
                                                                     atombox)
        hydronium_position = oxygen_frame[hydronium_index]
        if visualize:
            print(len(atoms[i]) + 1)
            print()
            for atom in atoms[i]:
                print(atom["name"], " ".join(map(str, atom["pos"])))
        print("S", " ".join(map(str, hydronium_position)), flush=True)
    print("Number of jumps:", hydronium_helper.jumps)


@argparse_compatible
def print_hydronium_and_solvationshell(trajectory, pbc, *, frame):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    with open(trajectory, "rb") as f:
        atomnumber = int(f.readline())
        f.seek(0)
        traj = xyz_parser.parse_xyz(f, atomnumber + 2, no_of_frames=frame + 1)
    oxygen_frame = traj["pos"][traj["name"] == b"O"]
    proton_frame = traj["pos"][traj["name"] == b"H"]

    hydronium_helper = HydroniumHelper()
    hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame, proton_frame,
                                                                 atombox)

    proton_dists = atombox.length_all_to_all(oxygen_frame[hydronium_index].reshape((-1, 3)),
                                             proton_frame).flatten()
    closest_four = np.argsort(proton_dists)[:4]
    oxygen_mask = np.zeros(oxygen_frame.shape[0], dtype=bool)
    oxygen_mask[hydronium_index] = 1
    oxygen_pos = oxygen_frame[hydronium_index]
    proton_mask = np.zeros(proton_frame.shape[0], dtype=bool)
    proton_mask[closest_four] = 1

    print(oxygen_frame.shape[0] + proton_frame.shape[0])
    print()
    print("O", *oxygen_frame[oxygen_mask][0])
    for proton_pos in proton_frame[proton_mask]:
        reduce_pbc(proton_pos, oxygen_pos, pbc)
        print("H", *proton_pos)
    for oxy in oxygen_frame[~oxygen_mask]:
        reduce_pbc(oxy, oxygen_pos, pbc)
        print("O", *oxy)
    for prot in proton_frame[~proton_mask]:
        reduce_pbc(prot, oxygen_pos, pbc)
        print("H", *prot)


@argparse_compatible
def show_kmc_rescaling(trajectory, pbc, *, rescaling_file, oxygen_index, frame):
    rescale_data = np.loadtxt(rescaling_file, usecols=(0, -1))
    rescale_func = interp1d(rescale_data[:, 0], rescale_data[:, -1], kind="linear")

    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    with open(trajectory, "rb") as f:
        atomnumber = int(f.readline())
        f.seek(0)
        traj = xyz_parser.parse_xyz(f, atomnumber + 2, no_of_frames=frame + 1)
    oxygen_frame = traj[frame]["pos"][traj[frame]["name"] == b"O"]
    proton_frame = traj[frame]["pos"][traj[frame]["name"] == b"H"]

    oo_dists = atombox.length_all_to_all(oxygen_frame.reshape((-1, 3)),
                                         oxygen_frame.reshape((-1, 3)))
    closest_three = np.argsort(oo_dists)[oxygen_index, 1:4]
    oxygen_mask = np.zeros(oxygen_frame.shape[0], dtype=bool)
    oxygen_mask[oxygen_index] = 1
    central_oxygen_pos = oxygen_frame[oxygen_index]
    neighbor_mask = np.zeros(oxygen_frame.shape[0], dtype=bool)
    neighbor_mask[closest_three] = 1

    h2o_groups = group_h2os(oxygen_frame, proton_frame, atombox)

    print(oxygen_frame.shape[0] + proton_frame.shape[0] + 9)
    print()
    first_h2o = oxygen_frame[oxygen_index], *proton_frame[h2o_groups[oxygen_index]]
    print_h2o(*first_h2o, pbc)

    for oxy_index in closest_three:
        old_pos = oxygen_frame[oxy_index]
        reduce_pbc(old_pos, central_oxygen_pos, pbc)
        distance = np.linalg.norm(old_pos - central_oxygen_pos)
        new_pos = rescale_distance(old_pos, central_oxygen_pos,
                                   rescale_interpolation_function(rescale_func, distance,
                                                                  rescale_func.x[0],
                                                                  rescale_func.x[-1],
                                                                  rescale_func.y[0]))
        diffvec = new_pos - old_pos
        print_h2o(new_pos, *(proton_frame[h2o_groups[oxy_index]] + diffvec), pbc)

    for oxy_index in closest_three:
        oxy_pos = oxygen_frame[oxy_index]
        reduce_pbc(oxy_pos, central_oxygen_pos, pbc)
        print_h2o(oxy_pos, *proton_frame[h2o_groups[oxy_index]], pbc)

    for oxy_index in np.arange(oxygen_frame.shape[0]):
        if oxy_index == oxygen_index or oxy_index in closest_three:
            continue
        oxy_pos = oxygen_frame[oxy_index]
        reduce_pbc(oxy_pos, central_oxygen_pos, pbc)
        print_h2o(oxy_pos, *proton_frame[h2o_groups[oxy_index]], pbc)


@argparse_compatible
def distance_histogram_between_hydronium_and_closest_oxygen(trajectory, pbc, dmin, dmax, bins, *,
                                                            plot=False):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_atoms(trajectory, "O", "H")
    hydronium_helper = HydroniumHelper()

    # distance_histogram = np.zeros(args.bins, dtype=int)
    distances = []

    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        if i % 1000 == 0:
            print(i, end="\r", flush=True)
        hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame, proton_frame,
                                                                     atombox)
        not_hydronium_index = np.arange(oxygen_frame.shape[0]) != hydronium_index
        closest_oxygen_index, distance = atombox.next_neighbor(oxygen_frame[hydronium_index],
                                                               oxygen_frame[not_hydronium_index])

        distances.append(distance)

    distance_count, edges = np.histogram(distances, bins=bins, range=(dmin, dmax))
    distance = (edges[:-1] + edges[1:]) / 2
    if plot:
        plt.plot(distance, distance_count)
        plt.show()

    for d, c in zip(distance, distance_count):
        print("{:10.4f} {:10.4f}".format(d, c))


@argparse_compatible
def distance_histogram_between_hydronium_and_all_oxygens(trajectory, pbc, dmin, dmax, bins, *,
                                                         plot=False, print_=False, clip=None,
                                                         normalized=False):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_atoms(trajectory, "O", "H")
    hydronium_helper = HydroniumHelper()

    if normalized:
        maxlen = np.sqrt(((pbc / 2)**2).sum())
        max_bins = int(maxlen / (dmax - dmin) * bins)
        range_ = (0, maxlen)
    else:
        range_ = (dmin, dmax)
        max_bins = bins

    histogram = np.zeros(max_bins, dtype=int)

    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        if clip and i == clip:
            break
        if i % 1000 == 0:
            print(i, end="\r", flush=True, file=sys.stderr)
        hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame, proton_frame,
                                                                     atombox)
        distances = atombox.length_all_to_all(oxygen_frame[[hydronium_index]], oxygen_frame[
            np.arange(oxygen_frame.shape[0]) != hydronium_index])

        histo, edges = np.histogram(distances, bins=max_bins, range=range_)
        histogram += histo
    print()

    distance = (edges[:-1] + edges[1:]) / 2

    if normalized:
        histogram = np.array(histogram, dtype=float) / histogram.sum() / (edges[1] - edges[0])

    mask = np.logical_and(dmin <= distance, distance <= dmax)

    if plot:
        plt.plot(distance[mask], histogram[mask])
        plt.show()

    if print_:
        for d, c in zip(distance, histogram):
            print(d, c)

    return {"distance": distance,
            "edges": edges,
            "histogram": histogram,
            "trajectory_length": oxygens.shape[0],
            "oxygen_number": oxygens.shape[1]
            }


@argparse_compatible
def rdf_between_hydronium_and_all_oxygens(trajectory, pbc, dmin, dmax, bins, *, plot=False,
                                          clip=None, print_=True):
    """Calculate the RDF between hydronium and neutral water molecules"""
    results = distance_histogram_between_hydronium_and_all_oxygens(trajectory, pbc, dmin, dmax,
                                                                   bins, clip=clip)
    distance, edges, histogram = results["distance"], results["edges"], results["histogram"]
    number_of_oxygens = results["oxygen_number"]
    trajectory_length = results["trajectory_length"] if not clip else clip
    histo_per_frame_and_particle = np.array(histogram, dtype=float) / trajectory_length
    rho = (number_of_oxygens - 1) / (pbc[0] * pbc[1] * pbc[2])
    distance_distribution_ideal_gas = 4. / 3 * np.pi * rho * (edges[1:]**3 - edges[:-1]**3)
    rdf = histo_per_frame_and_particle / distance_distribution_ideal_gas

    if plot:
        plt.plot(distance, rdf)
        plt.show()

    if print_:
        print("# Rho =", rho)
        print("# Distance, RDF")
        for d, r in zip(distance, rdf):
            print(d, r)

    return {"distance": distance,
            "edges": edges,
            "rdf": rdf,
            "histogram": histo_per_frame_and_particle
           }


def main(*args):
    """Determine the excess charge movement in a water box"""
    parser = argparse.ArgumentParser(
        description="Determine the excess charge movement in a water box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--visualize", action="store_true", help="Visualize excess charge position")
    parser.add_argument("--debug", action="store_true", help="Enable debug messages")
    parser.add_argument("trajectory", help="Trajectory path")
    parser.add_argument("pbc", nargs=3, type=float, help="periodic boundary conditions")

    subparsers = parser.add_subparsers(dest="subparser_name")

    parser_cv = subparsers.add_parser("cv", help="Track evolution of collective variable")
    parser_cv.set_defaults(func=track_collective_variable)

    parser_occupation = subparsers.add_parser("occupation", help="Track position of H3O+ ion")
    parser_occupation.set_defaults(func=track_hydronium_ion)

    parser_histo_closest = subparsers.add_parser("histo_closest",
                                                 help="Calculate distance histogram between "
                                                      "hydronium and the closest oxygen")
    parser_histo_closest.add_argument("--dmin", default=2.0, type=float,
                                      help="Minimum distance of histogram")
    parser_histo_closest.add_argument("--dmax", default=3.0, type=float,
                                      help="Maximum distance of histogram")
    parser_histo_closest.add_argument("--bins", default=50, type=int,
                                      help="Number of bins in histogram")
    parser_histo_closest.add_argument("--plot", action="store_true", help="Plot result")
    parser_histo_closest.set_defaults(func=distance_histogram_between_hydronium_and_closest_oxygen)

    parser_histo_all = subparsers.add_parser("histo_all",
                                             help="Calculate distance histogram between "
                                                  "hydronium and all oxygens")
    parser_histo_all.add_argument("--dmin", default=2.0, type=float,
                                  help="Minimum distance of histogram")
    parser_histo_all.add_argument("--dmax", default=3.0, type=float,
                                  help="Maximum distance of histogram")
    parser_histo_all.add_argument("--bins", default=50, type=int,
                                  help="Number of bins in histogram")
    parser_histo_all.add_argument("--plot", action="store_true", help="Plot result")
    parser_histo_all.add_argument("--normalized", action="store_true", help="Normalize histogram")
    parser_histo_all.add_argument("--clip", type=int, help="Determine maximum length")
    parser_histo_all.set_defaults(func=distance_histogram_between_hydronium_and_all_oxygens)

    parser_rdf = subparsers.add_parser("rdf", help="Calculate RDF between hydronium and the closest"
                                       "oxygen")
    parser_rdf.add_argument("--dmin", default=2.0, type=float, help="Minimum distance of histogram")
    parser_rdf.add_argument("--dmax", default=3.0, type=float, help="Maximum distance of histogram")
    parser_rdf.add_argument("--bins", default=50, type=int, help="Number of bins in histogram")
    parser_rdf.add_argument("--plot", action="store_true", help="Plot result")
    parser_rdf.add_argument("--clip", type=int, help="Determine maximum length")
    parser_rdf.set_defaults(func=rdf_between_hydronium_and_all_oxygens)

    parser_print = subparsers.add_parser("print_hydronium", help="Print hydronium structure")

    parser_print.add_argument("frame", type=int, help="Trajectory frame")
    parser_print.set_defaults(func=print_hydronium_and_solvationshell)

    parser_kmcrescale = subparsers.add_parser("rescale", help="Rescale neutral H2O trajectory")

    parser_kmcrescale.add_argument("rescaling_file", help="Rescaling file")
    parser_kmcrescale.add_argument("oxygen_index", type=int, help="Oxygen index")
    parser_kmcrescale.add_argument("frame", type=int, help="Frame")

    parser_kmcrescale.set_defaults(func=show_kmc_rescaling)

    args = parser.parse_args()

    if args.subparser_name == "histo_all":
        args.print_ = True

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    args.func(args)
