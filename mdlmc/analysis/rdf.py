import numpy as np
import argparse
import ipdb
import warnings
import matplotlib.pylab as plt
import time
from typing import List

from mdlmc.IO import xyz_parser
import mdlmc.atoms.numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.misc.tools import timer


def determine_histogram_one_atomtype(atomic_distances, **histo_kwargs):
    return np.histogram(np.triu(atomic_distances, k=1), **histo_kwargs)


def determine_histogram_two_atomtypes(atomic_distances, **histo_kwargs):
    return np.histogram(atomic_distances, **histo_kwargs)


def radial_distribution_function(atom_trajectories: List, atombox, histo_kwargs):
    histogram = np.zeros(histo_kwargs["bins"], dtype=int)
    if len(atom_trajectories) == 1:
        histo_fct = determine_histogram_one_atomtype
        trajectory_1, trajectory_2 = atom_trajectories[0], atom_trajectories[0]
    else:
        histo_fct = determine_histogram_two_atomtypes
        trajectory_1, trajectory_2 = atom_trajectories

    start_time = time.time()
    for i, (frame_1, frame_2) in enumerate(zip(trajectory_1, trajectory_2)):
        dists = atombox.length_all_to_all(frame_1, frame_2)
        if i % 1000 == 0:
            print("{:6d} ({:8.2f} fps)".format(i, float(i) / (time.time() - start_time)), end="\r",
                  flush=True)
            histo, edges = histo_fct(dists, **histo_kwargs)
            histogram += histo
    dists = (edges[:-1] + edges[1:]) / 2

    return histogram, dists, edges


def main(*args):
    parser = argparse.ArgumentParser(description="Calculates RDF")
    parser.add_argument("file", help="trajectory")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("--dmin", type=float, default=2.0, help="minimal value")
    parser.add_argument("--dmax", type=float, default=3.0, help="maximal value")
    parser.add_argument("--bins", type=int, default=50, help="number of bins")
    parser.add_argument("--clip", type=int, help="Clip trajectory after frame")
    parser.add_argument("-e", "--elements", nargs="+", default=["O"], help="Elements")
    parser.add_argument("-a", "--acidic_protons", action="store_true",
                        help="Only select acidic protons")
    args = parser.parse_args()

    histo_kwargs = {"range": (args.dmin, args.dmax),
                    "bins": args.bins}

    if len(args.elements) > 2:
        warnings.warn("Received more than two elements. Will just skip elements after the first two")

    trajectory = xyz_parser.load_atoms(args.file, clip=args.clip)
    pbc = np.array(args.pbc)
    atombox = AtomBoxCubic(pbc)

    if len(args.elements) > 2:
        raise ValueError("Too many elements specified")

    selection = npa.select_atoms(trajectory, args.elements)

    if args.acidic_protons:
        if "H" not in args.elements:
            raise ValueError("You specified acidic protons, but did not specify protons in "
                             "--elements")
        protons = selection[args.element.index("H")]
        acidic_proton_indices = npa.get_acidic_protons(trajectory[0], atombox)
        acidic_protons = protons[:, acidic_proton_indices]
        selection[args.element.index("H")] = acidic_protons

    histo, dists, edges = radial_distribution_function(selection, atombox, histo_kwargs)
    N = selection[-1].shape[1]
    V = pbc[0] * pbc[1] * pbc[2]
    rho = N / V

    print("# Number of particles:", N)
    print("# Volume:", V)
    print("# Rho = N / V =", rho)

    histo_norm = np.asfarray(histo) / (4. / 3 * np.pi * rho * (edges[1:]**3 - edges[:-1]**3)) / \
                 selection[-1].shape[1] / trajectory.shape[0]

    print("  Distance  Histogram        RDF")
    for d, h, hn in zip(dists, histo, histo_norm):
        print("{:10.8f} {:10.8f} {:10.8f}".format(d, h, hn))
