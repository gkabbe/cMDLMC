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
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a / (1 + np.exp((x - self.b) / self.c))

