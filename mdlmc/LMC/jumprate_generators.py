import numpy as np


class Fermi:
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
        if theta < np.pi / 2:
            return 0
        return super()(x)

