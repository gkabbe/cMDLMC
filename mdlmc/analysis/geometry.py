import sys
import numpy as np

from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic
from mdlmc.IO import trajectory_parser


def distance(a1, a2):
    return a2 - a1


def determine_neighbors():
    filename, x, y, z, atom_index, radius = sys.argv[1:]
    x, y, z = map(float, (x, y, z))
    pbc = np.array([x, y, z])
    atom_index = int(atom_index)
    radius = float(radius)
    trajectory = trajectory_parser.load_atoms(filename)
    atombox = AtomBoxCubic((x, y, z))

    center = trajectory[0][atom_index]

    neighbors = [center]
    for atom in trajectory[0]:
        if atombox.length(center["pos"], atom["pos"]) <= radius:
            neighbors.append(atom)

    print(len(neighbors))
    print()

    for atom in neighbors:
        while (distance(center["pos"], atom["pos"]) > pbc / 2).any():
            where = distance(center["pos"], atom["pos"]) > pbc / 2
            atom["pos"][where] -= pbc[where]
        while (distance(center["pos"], atom["pos"]) < -pbc / 2).any():
            where = distance(center["pos"], atom["pos"]) < -pbc / 2
            atom["pos"][where] += pbc[where]
        print(atom["name"].astype(str), " ".join(map(str, atom["pos"])))
