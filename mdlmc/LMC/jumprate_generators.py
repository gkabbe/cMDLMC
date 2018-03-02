from typing import Iterable

import numpy as np


def jumprate_generator(trajectory_generator: Iterable[np.array]):
    """
    Converts frames of a trajectory into arrays consisting of
    donor index, acceptor index, and jump probability

    Parameters
    ----------
    trajectory_generator: Iterable[np.array]

    Returns
    -------
    out: Iterable[np.array]
    """
    first_frame = next(trajectory_generator)

