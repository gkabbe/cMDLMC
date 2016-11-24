import numpy as np
import argparse
import matplotlib.pylab as plt
import time
from typing import List

from mdlmc.IO import xyz_parser
import mdlmc.atoms.numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.misc.tools import argparse_compatible


def distance_histogram(atom_trajectories: List, atombox, histo_kwargs):
    histogram = np.zeros(histo_kwargs["bins"], dtype=int)
    if len(atom_trajectories) == 1:
        trajectory_1, trajectory_2 = atom_trajectories[0], atom_trajectories[0]
        mask = np.ones((trajectory_1.shape[1], trajectory_1.shape[1]), dtype=bool)
        np.fill_diagonal(mask, 0)
    else:
        trajectory_1, trajectory_2 = atom_trajectories
        mask = slice(None, None)

    start_time = time.time()
    for i, (frame_1, frame_2) in enumerate(zip(trajectory_1, trajectory_2)):
        dists = atombox.length_all_to_all(frame_1, frame_2)
        if i % 1000 == 0:
            print("{:6d} ({:8.2f} fps)".format(i, float(i) / (time.time() - start_time)), end="\r",
                  flush=True)
        histo, edges = np.histogram(dists[mask], **histo_kwargs)
        histogram += histo
    dists = (edges[:-1] + edges[1:]) / 2

    return histogram, dists, edges


def calculate_rdf(atom_trajectories: List, atombox, histo_kwargs):
    histo, dists, edges = distance_histogram(atom_trajectories, atombox, histo_kwargs)

    if len(atom_trajectories) == 2:
        n1, n2 = atom_trajectories[0].shape[1], atom_trajectories[1].shape[1]
    else:
        n1, n2 = atom_trajectories[0].shape[1], atom_trajectories[0].shape[1] - 1

    volume = atombox.periodic_boundaries[0] * atombox.periodic_boundaries[1] * \
        atombox.periodic_boundaries[2]
    rho = n2 / volume
    trajectory_length = atom_trajectories[0].shape[0]

    histo_per_frame_and_particle = np.array(histo, dtype=float) / trajectory_length / n1
    distance_distribution_ideal_gas = 4. / 3 * np.pi * rho * (edges[1:]**3 - edges[:-1]**3)

    rdf = histo_per_frame_and_particle / distance_distribution_ideal_gas

    return rdf, dists


@argparse_compatible
def radial_distribution_function(file, pbc, elements, dmin, dmax, bins, *, clip=None, plot=False,
                                 acidic_protons=False, verbose=False):
    histo_kwargs = {"range": (dmin, dmax),
                    "bins": bins}

    trajectory = xyz_parser.load_atoms(file, clip=clip)
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)

    if len(elements) > 2:
        raise ValueError("Too many atom types specified")

    selection = npa.select_atoms(trajectory, *elements)

    if acidic_protons:
        if "H" not in elements:
            raise ValueError("You specified acidic protons, but did not specify protons in "
                             "--elements")
        acidic_proton_indices = npa.get_acidic_proton_indices(trajectory[0], atombox, verbose=verbose)
        acidic_protons = trajectory[:, acidic_proton_indices]
        acidic_protons = np.array(acidic_protons["pos"], order="C")
        selection[elements.index("H")] = acidic_protons

    rdf, dists = calculate_rdf(selection, atombox, histo_kwargs)

    if plot:
        plt.plot(dists, rdf)
        plt.show()

    print("{:10} {:10} {:10}".format("Distance", "RDF"))
    for d, r in zip(dists, rdf):
        print("{:10.8f} {:10.8f}".format(d, r))


@argparse_compatible
def calculate_distance_histogram(file, pbc, elements, dmin, dmax, bins, *, clip=None, plot=False,
                                 acidic_protons=False, normalized=False, verbose=False):
    if len(elements) > 2:
        raise ValueError("Too many atom types specified")

    trajectory = xyz_parser.load_atoms(file, clip=clip)
    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)

    if normalized:
        maxlen = np.sqrt(((pbc / 2)**2).sum())
        range_ = (0, maxlen)
        max_bins = int(maxlen / (dmax - dmin) * bins)
    else:
        range_ = (dmin, dmax)
        max_bins = bins

    if len(elements) > 2:
        raise ValueError("Too many elements specified")

    selection = npa.select_atoms(trajectory, *elements)

    if acidic_protons:
        if "H" not in elements:
            raise ValueError("You specified acidic protons, but did not specify protons in "
                             "--elements")
        acidic_proton_indices = npa.get_acidic_proton_indices(trajectory[0], atombox, verbose=verbose)
        acidic_protons = trajectory[:, acidic_proton_indices]
        acidic_protons = np.array(acidic_protons["pos"], order="C")
        selection[elements.index("H")] = acidic_protons

    histo, dists, edges = distance_histogram(selection, atombox, {"bins": max_bins, "range": range_})

    if normalized:
        histo = np.array(histo, dtype=float) / histo.sum() / (edges[1] - edges[0])

    mask = np.logical_and(dmin <= dists, dists <= dmax)

    if plot:
        plt.plot(dists[mask], histo[mask])
        plt.show()

    print("{:12} {:12}".format("Distance", "Probability"))
    for d, h in zip(dists[mask], histo[mask]):
        print("{:12.8f} {:12.8f}".format(d, h))


def main(*args):
    parser = argparse.ArgumentParser(description="Calculates distance histograms and RDF")
    parser.add_argument("file", help="trajectory")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins")
    parser.add_argument("--dmin", type=float, default=2.0, help="Minimal value")
    parser.add_argument("--dmax", type=float, default=3.0, help="Maximal value")
    parser.add_argument("--clip", type=int, help="Clip trajectory after frame")
    parser.add_argument("-e", "--elements", nargs="+", default=["O"], help="Elements")
    parser.add_argument("-a", "--acidic_protons", action="store_true",
                            help="Only select acidic protons")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--plot", action="store_true", help="Plot result")

    subparsers = parser.add_subparsers()

    parser_rdf = subparsers.add_parser("rdf", help="Determine radial distribution function")
    parser_rdf.add_argument("--dmin", type=float, default=2.0, help="Minimal value")
    parser_rdf.add_argument("--dmax", type=float, default=3.0, help="Maximal value")
    parser_rdf.set_defaults(func=radial_distribution_function)

    parser_histo = subparsers.add_parser("histo", help="Determine distance histogram")
    parser_histo.add_argument("--normalized", "-n", action="store_true",
                                 help="Normalize histogram")
    parser_histo.set_defaults(func=calculate_distance_histogram)

    args = parser.parse_args()

    args.func(args)
