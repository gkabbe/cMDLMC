from typing import Iterator

import numpy as np


def jumprate_generator(trajectory_generator: Iterator[np.array]):
    """
    Converts frames of a trajectory into arrays consisting of
    donor index, acceptor index, and jump probability

    Parameters
    ----------
    trajectory_generator: Iterator[np.array]

    Returns
    -------
    out: Iterator[np.array]
    """
    first_frame = next(trajectory_generator)

