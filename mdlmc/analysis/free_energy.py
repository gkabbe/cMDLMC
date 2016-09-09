import sys
import ipdb
import numpy as np
import pint
import matplotlib.pylab as plt

from mdkmc.IO import BinDump
from mdkmc.atoms import numpyatom as npa
from functools import reduce


ureg = pint.UnitRegistry()

k = 1.38064852e-23 * ureg.joule / ureg.kelvin
R = 8.3144621 * ureg.joule /  ureg.kelvin / ureg.mole
N_A = 6.022140857e23 / ureg.mole


def angle_vectorized(a1_pos, a2_pos, a3_pos, pbc):
    a2_a1= npa.distance(a2_pos, a1_pos, pbc)
    a2_a3 = npa.distance(a2_pos, a3_pos, pbc)
    return np.degrees(np.arccos(np.einsum("ij, ij -> i", a2_a1, a2_a3) / np.sqrt(np.einsum("ij,ij->i", a2_a1, a2_a1)) / np.sqrt(np.einsum("ij,ij->i", a2_a3, a2_a3))))


def potential_energy(dist):
    a, b, d_0 = 36 * ureg.kcal / ureg.mol / ureg.angstrom**2, -0.7 / ureg.angstrom**2, 2.22 * ureg.angstrom
    
    return a*(dist-d_0)/np.sqrt(b+1./(dist-d_0)**2)


def free_energy(dist, temp):
    return - k * temp * np.log(np.exp(-1./(R*temp) * (potential_energy(dist))).mean()), - k * temp * np.log(np.sqrt(np.exp(-1./(R*temp) * (potential_energy(dist))).var()))
    

def determine_PO_pairs(ox_traj, p_traj, pbc):
    P_neighbors = np.zeros(ox_traj.shape[1], int)

    for i, ox in enumerate(ox_traj[0]):
        P_index = npa.next_neighbor(ox, p_traj[0], pbc=pbc)[0]
        P_neighbors[i] = P_index
    return P_neighbors
    

def get_close_oxygen_indices(oxygen_traj, pbc):
    pairs = []
    for i, ox in enumerate(oxygen_traj[0]):
        dists = np.sqrt((npa.distance(ox, oxygen_traj[0, np.arange(oxygen_traj.shape[1]) != i], pbc) ** 2).sum(axis=-1))
        pair_index = np.argmin(dists)
        if pair_index >= i:
            pair_index += 1
        pairs.append((i, pair_index))
    pairs = [(i, j) for i, j in pairs if (j, i) in pairs and i < j]
    return pairs

def get_hbond_indices(oxygen_traj, proton_traj, pbc):
    pairs = []
    for i, prot in enumerate(proton_traj[0]):
        dists = np.sqrt((npa.distance(proton_traj[0, i], oxygen_traj[0], pbc) ** 2).sum(axis=-1))
        ox1, ox2 = np.argsort(dists)[:2]
        pairs.append((ox1, ox2))
    return pairs

def get_hbond_indices_all_traj(oxygen_traj, proton_traj, pbc):
    pairs = []
    for i, prot in enumerate(proton_traj[0]):
        dists = np.sqrt((npa.distance(proton_traj[:, None, i], oxygen_traj, pbc) ** 2).sum(axis=-1))
        ox1, ox2 = np.argsort(dists)[:, :2].T
        pairs.append((ox1, ox2))
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
        r_oh = np.sqrt((npa.distance(proton_traj[:, i], oxygen_traj[:, ox1], pbc) ** 2).sum(axis=-1))
        r_ho = np.sqrt((npa.distance(proton_traj[:, i], oxygen_traj[:, ox2], pbc) ** 2).sum(axis=-1))
        r_oo = np.sqrt((npa.distance(oxygen_traj[:, ox1], oxygen_traj[:, ox2], pbc) ** 2).sum(axis=-1))
        ipdb.set_trace()
    
    

def free_energy_from_oxygen_pairs(traj, pbc):
    oxygen_traj = npa.select_atoms(traj, "O")
    proton_traj = npa.select_atoms(traj, "H")
    pairs = get_hbond_indices(oxygen_traj, proton_traj, pbc)
    # Calculate distance between oxygen 0 and oxygen 1 over whole trajectory
    free_energies = []
    dists = []
    for i, j in pairs:
        d_oo = np.sqrt((npa.distance(oxygen_traj[:, i], oxygen_traj[:, j], pbc) ** 2).sum(axis=-1)) * ureg.angstrom
        d_oo = d_oo[d_oo <= 2.6 * ureg.angstrom]
        dists.append(d_oo)
        fe = free_energy(d_oo, 510. * ureg.kelvin)
        print(fe.to("kcal") * N_A)
        free_energies.append(fe)
        
    print("Free energy:", np.asfarray([(f.to("kcal")*N_A).magnitude for f in free_energies]).mean(), "kcal/mol")


def free_energy_standard_hbond_criterion(traj, pbc):
    oxygen_traj = npa.select_atoms(traj, "O")
    proton_traj = npa.select_atoms(traj, "H")
    
    chunk_size = 3000
    exp_val = 0
    dists = []
    for start, end in zip(range(0, traj.shape[0], chunk_size), range(chunk_size, traj.shape[0], chunk_size)):
        oh_bond_indices = get_hbond_indices_all_traj(oxygen_traj[start:end], proton_traj[start:end], pbc)
        for i in range(proton_traj.shape[1]):
            ox1 = oxygen_traj[range(start, end), oh_bond_indices[i][0]]
            ox2 = oxygen_traj[range(start, end), oh_bond_indices[i][1]]
            p = proton_traj[start:end, i]
            hb = is_hbond(ox1, ox2, p, pbc)
            if hb.any():
                dist = np.sqrt((npa.distance(ox1[hb], ox2[hb], pbc) ** 2).sum(axis=-1))
                dists += list(dist)   
        print(start)
            # print counter, ":", fe
    
    dists = np.array(dists) * ureg.angstrom
    result, error = free_energy(dists, 510*ureg.kelvin)
    print("Result for conventional geometric H-Bond criterion:")
    print(result.to("kcal") * N_A, error.to("kcal") * N_A)
    ipdb.set_trace()
    
    
def free_energy_mdkmc(traj, pbc):
    oxygen_traj = npa.select_atoms(traj, "O")
    phos_traj = npa.select_atoms(traj, "P")
    proton_traj = npa.select_atoms(traj, "H")
    p_neighbors = determine_PO_pairs(oxygen_traj, phos_traj, pbc)
   
    
    chunk_size = 3000
    exp_val = 0
    dists = []
    for start, end in zip(range(0, traj.shape[0], chunk_size), range(chunk_size, traj.shape[0], chunk_size)):
        oh_bond_indices = get_hbond_indices_all_traj(oxygen_traj[start:end], proton_traj[start:end], pbc)
        for i in range(proton_traj.shape[1]):
            ox1 = oxygen_traj[range(start, end), oh_bond_indices[i][0]]
            ox2 = oxygen_traj[range(start, end), oh_bond_indices[i][1]]
            phos = phos_traj[range(start, end), p_neighbors[oh_bond_indices[i][0]]]
            hb = is_hbond_mdkmc(ox1, ox2, phos, pbc)
            if hb.any():
                dist = np.sqrt((npa.distance(ox1[hb], ox2[hb], pbc) ** 2).sum(axis=-1))
                dists += list(dist)   
        print(start)
            # print counter, ":", fe
    
    dists = np.array(dists) * ureg.angstrom
    result, error = free_energy(dists, 510*ureg.kelvin)
    print("Result for POO geometric H-Bond criterion:")
    print(result.to("kcal") * N_A, error.to("kcal") * N_A)
    print(result.to("eV"), error.to("eV"))
    ipdb.set_trace()
    
    
def is_hbond(ox1, ox2, proton, pbc):
    return reduce(np.logical_and, [npa.angle_vectorized(ox1, proton, ox2, pbc) <= 60, np.linalg.norm(npa.distance(ox1, proton, pbc), axis=-1) <= 1.2,
                                   np.linalg.norm(npa.distance(ox2, proton, pbc), axis=-1) <= 2.2])


def is_hbond_mdkmc(ox1, ox2, phos, pbc):
    poo_angle = npa.angle_vectorized(phos, ox1, ox2, pbc)
    oo_dist = np.linalg.norm(npa.distance(ox1, ox2, pbc), axis=-1)

    return np.logical_and(poo_angle <= 90, oo_dist < 3.0)


def main(*args):
    try:
        pbc = np.asfarray([sys.argv[2], sys.argv[3], sys.argv[4]])
        traj = BinDump.npload_atoms(sys.argv[1], verbose=True)[:300000]
    except IndexError:
        print("Usage: {} <filename.xyz> <pbc_x> <pbc_y> <pbc_z>".format(sys.argv[0]))
        raise
    
    # free_energy_from_oxygen_pairs(traj, pbc)
    free_energy_standard_hbond_criterion(traj, pbc)
    # free_energy_mdkmc(traj, pbc)
    #  free_energy_when_proton_in_middle(traj, pbc)
