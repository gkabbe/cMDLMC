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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def linear_w_cutoff(x, a, b, d0):
    return np.where(x < d0, b, a * (x - d0) + b)


def normalize(OO_dist_interpolator, right_limit):
    xmin, xmax = OO_dist_interpolator.x[0], right_limit
    normalization_constant = quad(OO_dist_interpolator, xmin, right_limit, limit=100)
    return normalization_constant


def integrate_rdf(OO_dist_interpolator, up_to, number_of_atoms):
    xmin, xmax = OO_dist_interpolator.x[0], OO_dist_interpolator.x[-1]
    dist_fine = np.linspace(xmin, xmax, 1000)
    delta_d = dist_fine[1] - dist_fine[0]
    cumsum = np.cumsum(OO_dist_interpolator(dist_fine)) * number_of_atoms * delta_d
    idx = np.where(cumsum >= up_to)[0][0]
    return dist_fine[idx]


# noinspection PyTupleAssignmentBalance
def construct_conversion_fct(dist_histo, dist_histo_reference, number_of_atoms, fit_fct, p0=None):
    number_of_atoms_in_1st_solvation_shell = 3
    noa_1, noa_2 = number_of_atoms
    # Interpolate the distributions
    dist_histo_interpolator = interp1d(dist_histo[:, 0], dist_histo[:, 1], kind="cubic")
    dist_histo_reference_interpolator = interp1d(dist_histo_reference[:, 0],
                                                 dist_histo_reference[:, 1], kind="cubic")

    dist_histo_right_limit = integrate_rdf(dist_histo_interpolator,
                                           up_to=number_of_atoms_in_1st_solvation_shell,
                                           number_of_atoms=noa_1)
    dist_histo_reference_right_limit = integrate_rdf(dist_histo_reference_interpolator,
                                                     up_to=number_of_atoms_in_1st_solvation_shell,
                                                     number_of_atoms=noa_2)
    print(dist_histo_reference_right_limit, dist_histo_reference_right_limit)

    dist_fine = np.linspace(2.0, max(dist_histo_right_limit, dist_histo_reference_right_limit), 1000)

    delta_d = dist_fine[1] - dist_fine[0]

    cumsum = np.cumsum(dist_histo_interpolator(dist_fine)) * noa_1 * delta_d
    cumsum_reference = np.cumsum(dist_histo_reference_interpolator(dist_fine)) * noa_2 * delta_d

    right_limit = max(dist_histo_right_limit, dist_histo_reference_right_limit)
    # mask = reduce(np.logical_and, [dist_fine <= right_limit, cumsum > 10e-8])
    mask = cumsum > 1e-8

    left_limit = dist_fine[mask][0]

    # Define the conversion function.
    # For this, we search for each value in cumsum, where
    # its location in cumsum_reference would be
    # The result is a conversion function, which maps distances
    # of one distribution to distances of the other distribution
    convert = np.searchsorted(cumsum_reference[mask], cumsum[mask])

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

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col")
    # ax1.plot(dist_fine[mask], cumsum[mask], label="Classical MD")
    # ax1.plot(dist_fine[mask], cumsum_reference[mask], label="Hydronium - Oxygen AIMD")
    # # ax1.hlines(3, dist_fine[mask][0], dist_fine[-1])
    # ax1.legend(loc="upper left")
    # ax2.plot(dist_fine[mask], dist_fine[mask][convert], label="Conversion data")
    # ax2.plot(dist_fine, dist_fine, "g--")
    # ax2.plot(dist_fine[mask], fit_fct(dist_fine[mask], *popt), "r-", label="Fit")
    # ax2.legend(loc="upper left")

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
    args = parser.parse_args()

    neutral_pbc = np.array(args.neutral_pbc)
    atombox_neutral = AtomBoxCubic(neutral_pbc)
    reference_pbc = np.array(args.reference_pbc)

    f = h5py.File(args.neutral_trajectory, "r")
    neutral_traj = f[args.hdf5_key]

    # select all atoms (assuming that the trajectory only contains oxygens)
    selection = np.ones(neutral_traj.shape[1], dtype=bool)

    dists_neut, histo_neut = rdf.calculate_distance_histogram(neutral_traj, selection, selection,
                                                              atombox_neutral, dmin=2.0, dmax=4.0,
                                                              bins=100)

    dists_ref, histo_ref = rdf.prepare_trajectory(args.reference_trajectory, reference_pbc, bins=100,
                                                  dmin=2.0, dmax=4.0, clip=100000, elements=["O"],
                                                  acidic_protons=False, verbose=True, plot=True,
                                                  chunk_size=40000, normalized=True,
                                                  method="histo")

    results = excess_charge_analyzer.distance_histogram_between_hydronium_and_all_oxygens(
        args.reference_trajectory, reference_pbc, dmin=2.0, dmax=4, bins=100, plot=True,
        normalized=True, print_=True, clip=None)

    parameter_dict_md, plot_dict2 = construct_conversion_fct(
        histo_neut,
        histo_ref,
        number_of_atoms=(215, 99),
        fit_fct=linear_w_cutoff,
        p0=(1, 2.2, 2.5))

if __name__ == "__main__":
    main()
