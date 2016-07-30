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
    

def determine_PO_pairs(ox_traj, p_traj, pbc):
    P_neighbors = np.zeros(ox_traj.shape[1], int)

    for i, ox in enumerate(ox_traj[0]):
        P_index = npa.nextNeighbor(ox, p_traj[0], pbc=pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors
    

def get_close_oxygen_indices(oxygen_traj, pbc):
    pairs = []
    for i, ox in enumerate(oxygen_traj[0]):
        dists = np.sqrt((npa.distance_one_to_many(ox, oxygen_traj[0, np.arange(oxygen_traj.shape[1]) != i], pbc)**2).sum(axis=-1))
        pair_index = np.argmin(dists)
        if pair_index >= i:
            pair_index += 1
        pairs.append((i, pair_index))
    pairs = [(i, j) for i, j in pairs if (j, i) in pairs and i < j]
    return pairs

def get_hbond_indices(oxygen_traj, proton_traj, pbc):
    pairs = []
    for i, prot in enumerate(proton_traj[0]):
        dists = np.sqrt((npa.distance_one_to_many(proton_traj[0, i], oxygen_traj[0], pbc)**2).sum(axis=-1))
        ox1, ox2 = np.argsort(dists)[:, :2].T
        ipdb.set_trace()
        
        # Now check if they stay the same?
        
        # pairs.append((ox1, ox2))
    return pairs

def get_hbond_indices_all_traj(oxygen_traj, proton_traj, pbc):
    pairs = []
    for i, prot in enumerate(proton_traj[0]):
        dists = np.sqrt((npa.distance_one_to_many(proton_traj[:, None, i], oxygen_traj, pbc)**2).sum(axis=-1))
        ox1, ox2 = np.argsort(dists)[:, :2].T
        ipdb.set_trace()
        
        # Now check if they stay the same?
        
        # pairs.append((ox1, ox2))
    return pairs

def free_energy_when_proton_in_middle(traj, pbc):
    BinDump.mark_acidic_protons(traj, pbc, verbose=True)
    oxygen_traj, proton_traj, phos_traj = npa.select_atoms(traj, "O", "AH", "P")
    p_neighbors = determine_PO_pairs(oxygen_traj, phos_traj, pbc)
    original_ox_indices = npa.map_indices(traj[0], "O")
    original_h_indices = npa.map_indices(traj[0], "AH")
    
    # ox_pairs = get_close_oxygen_indices(oxygen_traj, pbc)
    oxygen_pairs = get_hbond_indices(oxygen_traj, proton_traj, pbc)
    
    for i, (ox1, ox2) in enumerate(oxygen_pairs):
        r_oh = np.sqrt((npa.distance_one_to_many(proton_traj[:, i], oxygen_traj[:, ox1], pbc)**2).sum(axis=-1))
        r_ho = np.sqrt((npa.distance_one_to_many(proton_traj[:, i], oxygen_traj[:, ox2], pbc)**2).sum(axis=-1))
        r_oo = np.sqrt((npa.distance_one_to_many(oxygen_traj[:, ox1], oxygen_traj[:, ox2], pbc)**2).sum(axis=-1))
        ipdb.set_trace()
    
    

def free_energy_from_oxygen_pairs(traj, pbc):
    oxygen_traj = npa.select_atoms(traj, "O")
    pairs = get_close_oxygen_indices(oxygen_traj, pbc)
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
        traj = BinDump.npload_atoms(sys.argv[1], verbose=True)[:300000]
    except IndexError:
        print "Usage: {} <filename.xyz> <pbc_x> <pbc_y> <pbc_z>".format(sys.argv[0])
        raise
    
    # free_energy_from_oxygen_pairs(traj, pbc)
    free_energy_when_proton_in_middle(traj, pbc)
