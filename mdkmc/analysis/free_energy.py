import sys
import ipdb
import numpy as np
import pint
import matplotlib.pylab as plt

from mdkmc.IO import BinDump
from mdkmc.atoms import numpyatom as npa


ureg = pint.UnitRegistry()

k = 1.38064852e-23 * ureg.joule / ureg.kelvin
R = 8.3144621 * ureg.joule /  ureg.kelvin / ureg.mole
N_A = 6.022140857e23 / ureg.mole

def potential_energy(dist):
    a, b, d_0 = 36 * ureg.kcal / ureg.mol / ureg.angstrom**2, -0.7 / ureg.angstrom**2, 2.22 * ureg.angstrom
    
    return a*(dist-d_0)/np.sqrt(b+1./(dist-d_0)**2)

def free_energy(dist, temp):
    return - k * temp * np.log(np.exp(-1./(R*temp) * (potential_energy(dist))).mean())


def free_energy_from_oxygen_pairs(traj, pbc):
    oxygen_traj = npa.select_atoms(traj, "O")

    pairs = []
    for i, ox in enumerate(oxygen_traj[0]):
        dists = np.sqrt((npa.distance_one_to_many(ox, oxygen_traj[0, np.arange(oxygen_traj.shape[1]) != i], pbc)**2).sum(axis=-1))
        pair_index = np.argmin(dists)
        if pair_index >= i:
            pair_index += 1
        pairs.append((i, pair_index))
    pairs = [(i, j) for i, j in pairs if (j, i) in pairs and i < j]
    # Calculate distance between oxygen 0 and oxygen 1 over whole trajectory
    free_energies = []
    dists = []
    for i, j in pairs:
        dists_over_traj = np.sqrt((npa.distance_one_to_many(oxygen_traj[:, i], oxygen_traj[:, j], pbc)**2).sum(axis=-1)) * ureg.angstrom
        dists.append(dists_over_traj)
        fe = free_energy(dists_over_traj, 510. * ureg.kelvin)
        print fe.to("kcal") * N_A
        free_energies.append(fe)

    ipdb.set_trace()


def main(*args):
    try:
        pbc = np.asfarray([sys.argv[2], sys.argv[3], sys.argv[4]])
        traj = BinDump.npload_atoms(sys.argv[1], verbose=True)
    except IndexError:
        print "Usage: {} <filename.xyz> <pbc_x> <pbc_y> <pbc_z>".format(sys.argv[0])
        raise
        
    free_energy_from_oxygen_pairs(traj, pbc)
    
