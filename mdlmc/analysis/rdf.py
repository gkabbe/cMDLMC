import argparse
import logging
import os
import time

import matplotlib.pylab as plt
import numpy as np

import mdlmc.atoms.numpyatom as npa
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.IO import xyz_parser
from mdlmc.misc.tools import argparse_compatible, chunk, chunk_trajectory


logger = logging.getLogger(__name__)


def calculate_histogram(traj_generator1, traj_generator2, atombox, dmin, dmax, bins, *,
                        normalized=False, verbose=False, mask=None):
    if normalized:
        pbc = np.array(atombox.periodic_boundaries)
        maxlen = np.sqrt(((pbc / 2)**2).sum())
        range_ = (0, maxlen)
        max_bins = int(maxlen / (dmax - dmin) * bins)
    else:
        range_ = (dmin, dmax)
        max_bins = bins
    histogram = np.zeros(max_bins)
    counter = 0
    start_time = time.time()
    for (_, end, chk1), (_, _, chk2) in zip(traj_generator1, traj_generator2):
        chk1 = np.array(chk1, order="C")
        chk2 = np.array(chk2, order="C")
        for frame_1, frame_2 in zip(chk1, chk2):
            dists = atombox.length_all_to_all(frame_1, frame_2)
            histo, edges = np.histogram(dists[mask], bins=max_bins, range=range_)
            if counter % 1000 == 0 and verbose:
                print("{:8d} ({:8.2f} fps)".format(counter,
                                                   float(counter) / (time.time() - start_time)),
                      end="\r", flush=True)
            histogram += histo
            counter += 1

    dists = (edges[:-1] + edges[1:]) / 2

    if normalized:
        histogram = np.array(histogram, dtype=float) / histogram.sum() / (edges[1] - edges[0])

    return dists, edges, histogram


def calculate_rdf(trajectory, selection1, selection2, atombox, dmin, dmax, bins, *, chunk_size=1,
                  clip=None, verbose=False):
    traj_gen1 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection1)
    traj_gen2 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection2)

    dists, edges, histo = calculate_histogram(traj_gen1, traj_gen2, atombox, dmin, dmax, bins,
                                              verbose=verbose)

    if (selection1 == selection2).all():
        n1, n2 = selection1.sum(), selection1.sum() - 1
    else:
        n1, n2 = selection1.sum(), selection2.sum()

    if verbose:
        print("# n1 =", n1)
        print("# n2 =", n2)

    volume = atombox.periodic_boundaries[0] * atombox.periodic_boundaries[1] * \
             atombox.periodic_boundaries[2]
    rho = n2 / volume
    trajectory_length = trajectory.shape[0] if not clip else clip

    if verbose:
        print("# Trajectory length:", trajectory_length)

    histo_per_frame_and_particle = np.array(histo, dtype=float) / (trajectory_length * n1)
    distance_distribution_ideal_gas = 4. / 3 * np.pi * rho * (edges[1:]**3 - edges[:-1]**3)

    rdf = histo_per_frame_and_particle / distance_distribution_ideal_gas

    return rdf, dists


def radial_distribution_function(trajectory, selection1, selection2, atombox, dmin, dmax,
                                 bins, *, clip=None, plot=False, verbose=False, chunk_size=1):
    rdf, dists = calculate_rdf(trajectory, selection1, selection2, atombox, dmin, dmax, bins,
                               chunk_size=chunk_size, clip=clip, verbose=verbose)

    if plot:
        plt.plot(dists, rdf)
        plt.show()

    return dists, rdf


def calculate_distance_histogram(trajectory, selection1, selection2, atombox, dmin, dmax, bins, *,
                                 clip=None, plot=False, normalized=False, verbose=False,
                                 chunk_size=1, single_element=False):
    traj_gen1 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection1)
    traj_gen2 = chunk_trajectory(trajectory, chunk_size, length=clip, selection=selection2)

    mask = np.ones((selection1.sum(), selection2.sum()), dtype=bool)
    if single_element:
        np.fill_diagonal(mask, 0)

    dists, _, histogram = calculate_histogram(traj_gen1, traj_gen2, atombox, dmin, dmax, bins,
                                              normalized=normalized, verbose=verbose,
                                              mask=mask)

    mask = np.logical_and(dmin <= dists, dists <= dmax)
    if plot:
        plt.plot(dists[mask], histogram[mask])
        plt.show()

    return dists, histogram


@argparse_compatible
def prepare_trajectory(filename, pbc, bins, dmin, dmax, clip, elements, acidic_protons, chunk_size,
                       method, hdf5_key="trajectory", normalized=False, verbose=False, plot=False):
    """
    Parameters
    ----------
    filename: str
        Trajectory filename
    pbc: array_like
        Periodic boundaries
    bins: int
        Number of histogram bins
    dmin: float
        Lower histogram boundary
    dmax: float
        Upper histogram boundary
    clip: int
        Limit evaluation of trajectory to <clip> frames
    elements: List of strings
        Elements between which the histogram/RDF will be calculated
    acidic_protons: bool
        Whether to use all protons (acidic_protons = False) or only those whose next neighbor
        is an oxygen atom
    chunk_size: int
    method: str
        Choose between "rdf" and "histo"
    normalized: bool
        Normalize histogram
    verbose: bool
        Verbosity
    plot: bool
        Plot result

    Returns
    -------

    """
    _, ext = os.path.splitext(filename)
    if ext == ".hdf5":
        atom_names, trajectory = xyz_parser.load_trajectory_from_hdf5(filename, hdf5_key=hdf5_key)
    else:
        trajectory = xyz_parser.load_atoms(filename, clip=clip)
        atom_names = trajectory[0]["name"].astype(str)
        trajectory = np.array(trajectory["pos"], order="C")

    pbc = np.array(pbc)
    atombox = AtomBoxCubic(pbc)

    # Check if elements are actually contained in trajectory
    for element in elements:
        if element not in atom_names:
            raise ValueError("Element {} was not found in trajectory".format(element))

    logger.info("Determining the {} between {} and {}".format(
        "RDF" if method == "rdf" else "distance histogram", elements[0], elements[-1]))

    if acidic_protons:
        if "H" in elements:
            elements.remove("H")
        acid_indices = npa.get_acidic_proton_indices(trajectory[0], atombox, verbose=verbose)
        selection1 = np.zeros(trajectory.shape[1], dtype=bool)
        selection1[acid_indices] = 1
        if elements:
            selection2 = atom_names == elements[0]
            single_element = False
        else:
            selection2 = selection1
            single_element = True
    else:
        selection1 = atom_names == elements[0]
        elements.pop(0)

        if elements:
            selection2 = atom_names == elements[0]
            single_element = False
        else:
            selection2 = selection1
            single_element = True

    if method == "rdf":
        dists, hist = radial_distribution_function(trajectory, selection1, selection2, atombox,
                                                   dmin, dmax, bins, clip=clip, plot=plot,
                                                   verbose=verbose, chunk_size=chunk_size)
    elif method == "histo":
        dists, hist = calculate_distance_histogram(trajectory, selection1, selection2, atombox,
                                                   dmin, dmax, bins, clip=clip, plot=plot,
                                                   verbose=verbose, chunk_size=chunk_size,
                                                   normalized=normalized,
                                                   single_element=single_element)
    else:
        raise RuntimeError("What is", method, "?")

    print("# {:>10} {:>12}".format("Distance", "Probability"))
    for d, h in zip(dists, hist):
        print("{:12.8f} {:12.8f}".format(d, h))

    return dists, hist


def main(*args):
    parser = argparse.ArgumentParser(description="Calculates distance histograms and RDF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="Trajectory filename")
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
    parser.add_argument("--chunk_size", "-c", type=int, default=1,
                        help="Read trajectory in chunks ofsize <chunk_size>")
    parser.add_argument("--log", "-l", default="info", help="Set log level")

    subparsers = parser.add_subparsers(dest="method")

    parser_rdf = subparsers.add_parser("rdf", help="Determine radial distribution function")
    parser_rdf.set_defaults(func=prepare_trajectory)

    parser_histo = subparsers.add_parser("histo", help="Determine distance histogram")
    parser_histo.add_argument("--normalized", "-n", action="store_true",
                              help="Normalize histogram")
    parser_histo.set_defaults(func=prepare_trajectory)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()))

    args.func(args)
