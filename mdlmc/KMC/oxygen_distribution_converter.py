from functools import reduce
from inspect import signature
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pylab as plt


def find_first_solvation_shell(dist_histo, n=4):
    """Find the first solvation shell by integrating the distance probability over space.
    The point at which the integral equals the coordination number, the first solvation shell ends.
    """
    dist, prob = dist_histo.T


def normalize(OO_dist_interpolator, right_limit):
    xmin, xmax = OO_dist_interpolator.x[0], right_limit
    normalization_constant = quad(OO_dist_interpolator, xmin, right_limit, limit=100)
    return normalization_constant


def integrate_rdf(OO_dist_interpolator, up_to, number_of_atoms):
    xmin, xmax = OO_dist_interpolator.x[0], OO_dist_interpolator.x[-1]
    dist_fine = np.linspace(xmin, xmax, 1000)
    cumsum = np.cumsum(OO_dist_interpolator(dist_fine))
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
                                           number_of_atoms=number_of_atoms)
    dist_histo_reference_right_limit = integrate_rdf(dist_histo_reference_interpolator,
                                                     up_to=number_of_atoms_in_1st_solvation_shell,
                                                     number_of_atoms=number_of_atoms)
    print(dist_histo_reference_right_limit, dist_histo_reference_right_limit)

    dist_fine = np.linspace(2.0, max(dist_histo_right_limit, dist_histo_reference_right_limit), 1000)

    delta_d = dist_fine[1] - dist_fine[0]

    cumsum = np.cumsum(dist_histo_interpolator(dist_fine)) * noa_1
    cumsum_reference = np.cumsum(dist_histo_reference_interpolator(dist_fine)) * noa_2

    right_limit = max(dist_histo_right_limit, dist_histo_reference_right_limit)
    mask = reduce(np.logical_and, [dist_fine <= right_limit, cumsum > 10e-8])

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

    for k, v in zip(param_names[1:], popt):
        print("{} = {}".format(k, v))
        parameter_dict[k] = v
    parameter_dict["left_bound"] = left_limit
    parameter_dict["right_bound"] = right_limit
    print("{} < d < {}".format(left_limit, right_limit))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col")
    ax1.plot(dist_fine[mask], cumsum[mask] * delta_d, label="cumsum")
    ax1.plot(dist_fine[mask], cumsum_reference[mask] * delta_d, label="reference")
    ax1.legend(loc="upper left")
    ax2.plot(dist_fine[mask], dist_fine[mask][convert], label="Conversion data")
    ax2.plot(dist_fine, dist_fine, "g--")
    ax2.plot(dist_fine[mask], fit_fct(dist_fine[mask], *popt), "r-", label="Fit")
    ax2.legend(loc="upper left")

    return parameter_dict, fig
