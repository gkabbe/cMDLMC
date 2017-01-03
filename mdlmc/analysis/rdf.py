import argparse
import os
import time

import matplotlib.pylab as plt
import numpy as np

import mdlmc.atoms.numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.IO import xyz_parser
from mdlmc.misc.tools import argparse_compatible, chunk, chunk_trajectory


def distance_histogram(trajectory_1, trajectory_2, atombox, histo_kwargs, verbose=False):
    histogram = np.zeros(histo_kwargs["bins"], dtype=int)

    start_time = time.time()
    for i, (frame_1, frame_2) in enumerate(zip(trajectory_1, trajectory_2)):
        dists = atombox.length_all_to_all(frame_1, frame_2)
        if i % 1000 == 0 and verbose:
            print("{:6d} ({:8.2f} fps)".format(i, float(i) / (time.time() - start_time)), end="\r",
                  flush=True)
        histo, edges = np.histogram(dists, **histo_kwargs)
        histogram += histo
    dists = (edges[:-1] + edges[1:]) / 2

    return histogram, dists, edges


def calculate_rdf(trajectory, selection1, selection2, atombox, histo_kwargs):

    trj_gen1 = chunk_trajectory(trajectory, )

    histo, dists, edges = distance_histogram(trajectory, selection1, selection2, atombox,
                                             histo_kwargs)

    if (selection1 == selection2).all():
        n1, n2 = selection1.shape[0], selection1.shape[0] - 1
    else:
        n1, n2 = selection1.shape[1], selection2.shape[1]

    volume = atombox.periodic_boundaries[0] * atombox.periodic_boundaries[1] * \
        atombox.periodic_boundaries[2]
    rho = n2 / volume
    trajectory_length = trajectory[0].shape[0]

    histo_per_frame_and_particle = np.array(histo, dtype=float) / trajectory_length / n1
    distance_distribution_ideal_gas = 4. / 3 * np.pi * rho * (edges[1:]**3 - edges[:-1]**3)

    rdf = histo_per_frame_and_particle / distance_distribution_ideal_gas

    return rdf, dists


@argparse_compatible
def radial_distribution_function(trajectory, selection1, selection2, atombox, dmin, dmax,
                                 bins, *, clip=None, plot=False, verbose=False, chunk_size=None):

    histo_kwargs = {"range": (dmin, dmax),
                    "bins": bins}

    rdf, dists = calculate_rdf(trajectory, selection1, selection2, atombox, histo_kwargs)

    if plot:
        plt.plot(dists, rdf)
        plt.show()

    print("{:10} {:10} {:10}".format("Distance", "RDF"))
    for d, r in zip(dists, rdf):
        print("{:10.8f} {:10.8f}".format(d, r))


def calculate_distance_histogram(trajectory, selection1, selection2, atombox, dmin, dmax, bins, *,
                                 clip=None, plot=False, normalized=False, verbose=False,
                                 chunk_size=None):
    if not chunk_size:
        chunk_size = trajectory.shape[0]
        verbose_histo = verbose
    else:
        verbose_histo = False

    if normalized:
        pbc = np.array(atombox.periodic_boundaries)
        maxlen = np.sqrt(((pbc / 2)**2).sum())
        range_ = (0, maxlen)
        max_bins = int(maxlen / (dmax - dmin) * bins)
    else:
        range_ = (dmin, dmax)
        max_bins = bins

    traj_gen1 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection1)
    traj_gen2 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection2)

    histogram = np.zeros(max_bins)

    counter = 0
    start_time = time.time()
    for (_, end, chk1), (_, _, chk2) in zip(traj_gen1, traj_gen2):
        chk1 = np.array(chk1, order="C")
        chk2 = np.array(chk1, order="C")
        for frame_1, frame_2 in zip(chk1, chk2):
            dists = atombox.length_all_to_all(frame_1, frame_2)
            histo, edges = np.histogram(dists, bins=max_bins, range=range_)
            if counter % 1000 == 0 and verbose:
                print("{:8d} ({:8.2f} fps)".format(counter,
                                                    float(counter) / (time.time() - start_time)),
                      end="\r", flush=True)
            histogram += histo
            counter += 1

    dists = (edges[:-1] + edges[1:]) / 2

    if normalized:
        histogram = np.array(histogram, dtype=float) / histogram.sum() / (edges[1] - edges[0])
    mask = np.logical_and(dmin <= dists, dists <= dmax)

    if plot:
        plt.plot(dists[mask], histogram[mask])
        plt.show()

    print("{:12} {:12}".format("Distance", "Probability"))
    for d, h in zip(dists[mask], histogram[mask]):
        print("{:12.8f} {:12.8f}".format(d, h))


def prepare_trajectory(args):
    _, ext = os.path.splitext(args.file)
    if ext == ".hdf5":
        atom_names, trajectory = xyz_parser.load_trajectory_from_hdf5(args.file)
    else:
        trajectory = xyz_parser.load_atoms(args.file, clip=args.clip)
        atom_names = trajectory[0]["name"]
        trajectory = np.array(trajectory["pos"], order="C")

    pbc = np.array(args.pbc)
    atombox = AtomBoxCubic(pbc)

    if args.acidic_protons and "H" in args.elements:
        args.elements.remove("H")
        acid_indices = npa.get_acidic_proton_indices(trajectory[0], atombox, verbose=args.verbose)
        selection1 = np.zeros(trajectory.shape[1], dtype=bool)
        selection1[acid_indices] = 1
    else:
        selection1 = atom_names == args.elements[0]
        args.elements.pop(0)

    if args.elements and args.elements[0] != "H":
        selection2 = atom_names == args.elements[0]
    else:
        selection2 = selection1

    if args.subparser_name == "rdf":
        radial_distribution_function(trajectory, selection1, selection2, atombox, args.dmin,
                                     args.dmax, args.bins, clip=args.clip, plot=args.plot,
                                     verbose=args.verbose, chunk_size=args.chunk_size)
    elif args.subparser_name == "histo":
        calculate_distance_histogram(trajectory, selection1, selection2, atombox, args.dmin,
                                     args.dmax, args.bins, clip=args.clip, plot=args.plot,
                                     verbose=args.verbose, chunk_size=args.chunk_size,
                                     normalized=args.normalized)
    else:
        raise RuntimeError("What is", args.subparser_name, "?")


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
    parser.add_argument("--chunk_size", "-c", type=int, help="Read trajectory in chunks of"
                                                             "size <chunk_size>")

    subparsers = parser.add_subparsers(dest="subparser_name")

    parser_rdf = subparsers.add_parser("rdf", help="Determine radial distribution function")
    parser_rdf.add_argument("--dmin", type=float, default=2.0, help="Minimal value")
    parser_rdf.add_argument("--dmax", type=float, default=3.0, help="Maximal value")
    parser_rdf.set_defaults(func=prepare_trajectory)

    parser_histo = subparsers.add_parser("histo", help="Determine distance histogram")
    parser_histo.add_argument("--normalized", "-n", action="store_true",
                                 help="Normalize histogram")
    parser_histo.set_defaults(func=prepare_trajectory)

    args = parser.parse_args()

    args.func(args)
