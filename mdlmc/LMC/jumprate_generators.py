from typing import Iterator

import numpy as np


def jumprate_generator_distance(trajectory_generator: Iterator[np.array]):
    """
    Converts frames of a trajectory into arrays consisting of
    donor index, acceptor index, and jump probability.
    In this implementation the jump rate depends only on the donor-acceptor
    distance.

    Parameters
    ----------
    trajectory_generator: Iterator[np.array]
        Trajectory of donor and acceptor positions

    Returns
    -------
    out: Iterator[np.array]
    """
    first_frame = next(trajectory_generator)

