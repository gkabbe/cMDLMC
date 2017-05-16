import argparse
from functools import reduce
from inspect import signature
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pylab as plt
import tables  # needed for hdf5 compression
import h5py

from mdlmc.analysis import rdf, excess_charge_analyzer
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic


# Setup logger
logger = logging.getLogger(__name__)


def linear_w_cutoff(x, a, b, d0):
    return np.where(x < d0, b, a * (x - d0) + b)


def integrate_rdf(OO_dist_interpolator, up_to, number_of_atoms):
    xmin, xmax = OO_dist_interpolator.x[0], OO_dist_interpolator.x[-1]
    dist_fine = np.linspace(xmin, xmax, 1000)
    delta_d = dist_fine[1] - dist_fine[0]
    cumsum = np.cumsum(OO_dist_interpolator(dist_fine)) * number_of_atoms * delta_d
    idx = np.where(cumsum >= up_to)[0][0]
    return dist_fine[idx]


def construct_conversion_fct(dist_histo, dist_histo_reference, *, number_of_atoms, fit_fct,
                             noa_in_1st_solvationshell, p0=None, plot=False):
    """
    Determine the conversion function, which maps the H2O - H2O distance distribution (measured
    as the oxygen - oxygen distance) of an uncharged system to the H3O+ - H2O distance of a
    protonated target system.
    For this, the integrated H2O-H2O (H2O-H3O+) RDFs of the neutral (protonated) system are
    calculated.
    For each value of the integrated RDF of the neutral system, the distance is determined, at
    which the integrated RDF of the protonated system has the same value.
    The result is a conversion function, which maps the distance distribution of the neutral
    system to the distance distribution of the protonated system.

    Parameters
    ----------
    dist_histo: array_like
        Histogram of the H2O-H2O distances in the neutral system
    dist_histo_reference: array_like
        Histogram of the H2O-H3O+ distances in the protonated system
    number_of_atoms: tuple of int
        Total number of oxygens for neutral and protonated system
    fit_fct: Function
        Function which will be used to fit the resulting conversion function
    noa_in_1st_solvationshell: int
        Number of atoms in first solvation shell
    p0: tuple of Union[int, float]
        Initial fit parameters
    plot: bool
        Plots the resulting conversion function if set to True

    Returns
    -------
    parameter_dict: Dict
        Dictionary containing fit parameters and range of the fit function
    array_dict: Dict
        Dictionary containing the integrated RDFs, the conversion function, and the
        conversion fit
    """

    noa_1, noa_2 = number_of_atoms
    # Interpolate the distributions
    dist_histo_interpolator = interp1d(dist_histo[:, 0], dist_histo[:, 1], kind="cubic")
    dist_histo_reference_interpolator = interp1d(dist_histo_reference[:, 0],
                                                 dist_histo_reference[:, 1], kind="cubic")

    dist_histo_right_limit = integrate_rdf(dist_histo_interpolator,
                                           up_to=noa_in_1st_solvationshell,
                                           number_of_atoms=noa_1)
    dist_histo_reference_right_limit = integrate_rdf(dist_histo_reference_interpolator,
                                                     up_to=noa_in_1st_solvationshell,
                                                     number_of_atoms=noa_2)

    logger.debug("Right edge of reference histogram: {}".format(dist_histo_reference_right_limit))
    logger.debug("Right edge of neutral system histogram: {}".format(dist_histo_right_limit))

    left_limit = max(dist_histo[0, 0], dist_histo_reference[0, 0])
    right_limit = max(dist_histo_right_limit, dist_histo_reference_right_limit)
    dist_fine = np.linspace(left_limit, right_limit, 1000)

    delta_d = dist_fine[1] - dist_fine[0]

    cumsum = np.cumsum(dist_histo_interpolator(dist_fine)) * noa_1 * delta_d
    cumsum_reference = np.cumsum(dist_histo_reference_interpolator(dist_fine)) * noa_2 * delta_d

    # mask = reduce(np.logical_and, [dist_fine <= right_limit, cumsum > 10e-8])
    mask = cumsum > 1e-8

    left_limit = dist_fine[mask][0]

    # Define the conversion function.
    convert = np.searchsorted(cumsum_reference[mask], cumsum[mask])
    convert = np.where(convert < dist_fine[mask].size, convert, -1)

    if plot:
        plt.plot(dist_fine[mask], dist_fine[mask][convert], "x")
        plt.plot(dist_fine[mask], dist_fine[mask])
        plt.xlim(2, 3)
        plt.show()

    # Now fit the result
    popt, pcov = curve_fit(fit_fct, dist_fine[mask], dist_fine[mask][convert], p0=p0)

    param_names = list(signature(fit_fct).parameters.keys())
    parameter_dict = {}

    for k, v, v_err in zip(param_names[1:], popt, pcov.diagonal()):
        print("{} = {} +- {}".format(k, v, v_err))
        parameter_dict[k] = v
    parameter_dict["left_bound"] = left_limit
    parameter_dict["right_bound"] = right_limit
    print("{} < d < {}".format(left_limit, right_limit))

    array_dict = {
        "distance": dist_fine[mask],
        "RDFInt": cumsum[mask],
        "RDFIntReference": cumsum_reference[mask],
        "Conversion": dist_fine[mask][convert],
        "ConversionFit": fit_fct(dist_fine[mask], *popt)
    }

    return parameter_dict, array_dict


def find_first_solvation_shell_end(dists, dist_prob, dists2, rdf, number_of_atoms,
                                   noa_in_first_solvation_shell):
    xmin, xmax = 2.2, 3.5
    delta_d = dists[1] - dists[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    dist_prob_integrated = delta_d * np.cumsum(dist_prob) * number_of_atoms
    solvation_shell_end_index = \
        np.where(dist_prob_integrated >= noa_in_first_solvation_shell)[0].min()

    idx2 = np.where(dists2 >= dists[solvation_shell_end_index])[0][0]

    ax1.plot(dists, dist_prob_integrated, label="O--O AIMD")
    ax1.vlines(dists[solvation_shell_end_index], 0, dist_prob_integrated[solvation_shell_end_index],
               colors="g", linestyles="--")
    ax1.hlines(noa_in_first_solvation_shell, xmin, dists[solvation_shell_end_index], colors="g",
               linestyles="--")
    ax1.set_ylim((0, 5))
    ax1.set_ylabel("Integrated RDF")
    ax2.plot(dists2[dists2 < dists[solvation_shell_end_index] + 1],
             rdf[dists2 < dists[solvation_shell_end_index] + 1])
    ax2.vlines(dists[solvation_shell_end_index], 0, rdf.max(), colors="g", linestyles="--")
    ax2.set_ylabel("RDF")
    plt.xlabel("$d$ / \AA")
    plt.xlim((xmin, xmax))

    return dists[solvation_shell_end_index], fig


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("neutral_trajectory", help="File name of neutral water trajectory")
    parser.add_argument("neutral_pbc", nargs=3, type=float,
                        help="Periodic boundaries of neutral trajectory")
    parser.add_argument("reference_trajectory", help="File name of protonated reference water "
                                                     "trajectory")
    parser.add_argument("reference_pbc", nargs=3, type=float,
                        help="Periodic boundaries of reference trajectory")
    parser.add_argument("--hdf5_key", default="oxygen_trajectory", help="Array key in hdf5 file")
    parser.add_argument("--clip", type=int, default=None, help="Stop after <clip> frames")
    parser.add_argument("--log", "-l", default="info", help="Set log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()))

    neutral_pbc = np.array(args.neutral_pbc)
    atombox_neutral = AtomBoxCubic(neutral_pbc)
    reference_pbc = np.array(args.reference_pbc)

    f = h5py.File(args.neutral_trajectory, "r")
    neutral_traj = f[args.hdf5_key]

    # select all atoms (assuming that the trajectory only contains oxygens)
    selection = np.ones(neutral_traj.shape[1], dtype=bool)

    logger.info("Calculate distance histogram of neutral trajectory {}".format(args.neutral_trajectory))
    dists_neut, histo_neut = rdf.calculate_distance_histogram(neutral_traj, selection, selection,
                                                              atombox_neutral, dmin=2.0, dmax=4.0,
                                                              bins=100, clip=args.clip, verbose=True,
                                                              chunk_size=1000, plot=False,
                                                              normalized=True)

    logger.info("Calculate H3O+ H2O distance histogram of reference trajectory {}".format(
        args.reference_trajectory))
    results = excess_charge_analyzer.distance_histogram_between_hydronium_and_all_oxygens(
        args.reference_trajectory, reference_pbc, dmin=2.0, dmax=4, bins=100, plot=False,
        normalized=True, print_=True, clip=args.clip)

    dists_ref, histo_ref = results["distance"], results["histogram"]

    histo_neut = np.vstack([dists_neut, histo_neut]).T
    histo_ref = np.vstack([dists_ref, histo_ref]).T

    parameter_dict_md, plot_dict2 = construct_conversion_fct(
        histo_neut,
        histo_ref,
        number_of_atoms=(215, 99),
        fit_fct=linear_w_cutoff,
        p0=(1, 2.2, 2.5))

if __name__ == "__main__":
    main()
