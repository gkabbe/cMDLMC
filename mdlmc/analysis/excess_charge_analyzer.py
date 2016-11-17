import argparse
import matplotlib.pylab as plt
import numpy as np

from mdlmc.IO import xyz_parser
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic
from mdlmc.misc.tools import argparse_compatible


def excess_charge_collective_variable(oxygen_pos, proton_pos):
    return proton_pos.sum(axis=0) - 2 * oxygen_pos.sum(axis=0)


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


def main(*args):
    """Determine the excess charge movement in a water box"""
    parser = argparse.ArgumentParser(
        description="Determine the excess charge movement in a water box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--visualize", action="store_true", help="Visualize excess charge position")
    parser.add_argument("trajectory", help="Trajectory path")
    parser.add_argument("pbc", nargs=3, type=float, help="periodic boundary conditions")

    subparsers = parser.add_subparsers()

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
    parser_histo_all.set_defaults(func=distance_histogram_between_hydronium_and_all_oxygens)

    args = parser.parse_args()
    args.func(args)


@argparse_compatible
def track_collective_variable(trajectory, pbc, *, visualize=False):
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_atoms(trajectory, "O", "H")
    hydronium_helper = HydroniumHelper()
    excess_charge_start_index = hydronium_helper.determine_hydronium_index(oxygens[0],
                                                                              protons[0],
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
        hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame,
                                                                        proton_frame, atombox)
        closest_oxygen_index, distance = atombox.next_neighbor(oxygen_frame[hydronium_index],
                                                               oxygen_frame[np.arange(
                                                                   oxygen_frame.shape[
                                                                       0]) != hydronium_index])

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
                                                         plot=False, normalized=False):
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
        if i % 1000 == 0:
            print(i, end="\r", flush=True)
        hydronium_index = hydronium_helper.determine_hydronium_index(oxygen_frame,
                                                                        proton_frame, atombox)
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

    for d, c in zip(distance[mask], histogram[mask]):
        print(d, c)
