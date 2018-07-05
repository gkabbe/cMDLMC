# coding=utf-8

from abc import ABCMeta

import numpy as np


class JumpRate(metaclass=ABCMeta):
    """Calculates a proton hopping rate as a function of geometric parameters such as distance,
    angle, etc."""
    pass


class Fermi(JumpRate):

    __show_in_config__ = True

    def __init__(self, a: float, b: float, c: float):
        """
        Parameters
        ----------
        a:
            Amplitude
        b:
            Location
        c:
            Width
        """
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, x):
        return self._a / (1 + np.exp((x - self._b) / self._c))


class FermiAngle(Fermi):
    def __init__(self, a: float, b: float, c: float, theta: float):
        super().__init__(a, b, c)
        self._theta = theta

    def __call__(self, x, theta):
        return np.where(theta < self._theta, 0, super().__call__(x))

