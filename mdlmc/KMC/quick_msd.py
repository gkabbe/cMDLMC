import argparse

import numpy as np

from mdlmc.analysis.msd import displacement


def main():
    pbc = np.array(map(float, sys.argv[2:5]))
    data = np.loadtxt(sys.argv[1], usecols=(2, 3, 4))[:, None, :]
    displ = displacement(data, pbc)
