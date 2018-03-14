from itertools import tee
import sys
from typing import Iterator
import numpy as np
import logging

from scipy.sparse import lil_matrix

from mdlmc.cython_exts.LMC.PBCHelper import AtomBox


logger = logging.getLogger(__name__)


dtype_xyz = np.dtype([("name", np.str_, 2), ("pos", np.float64, (3,))])
dtype_xyz_bytes = np.dtype([("name", np.string_, 2), ("pos", np.float64, (3,))])

atom_masses = {'C' : 12.001,
               'Cl': 35.45,
               'Cs': 132.90545196,
               'H' : 1.008,
               'O' : 15.999,
               'P' : 30.973761998,
               'S' : 32.06,
               'Se': 78.971}


def get_acidic_proton_indices(frame, atom_box, verbose=False):
    """Returns the indices of all protons whose closest neighbor is an oxygen atom"""
    if len(frame.shape) > 1:
        raise ValueError("Argument frame should be a one dimensional numpy array of type dtype_xyz")
    acidic_indices = []
    atom_names = np.array(frame["name"], dtype=str)
    H_atoms = np.array(frame["pos"][atom_names == "H"])
    H_indices = np.where(atom_names == "H")[0]
    not_H_atoms = np.array(frame["pos"][atom_names != "H"])
    not_H_atoms_names = atom_names[atom_names != "H"]
    for i, H in enumerate(H_atoms):
        nn_index, next_neighbor = atom_box.next_neighbor(H, not_H_atoms)
        if not_H_atoms_names[nn_index] == "O":
            acidic_indices.append(H_indices[i])
    if verbose:
        print("# Acidic indices: ", acidic_indices)
        print("# Number of acidic protons: ", len(acidic_indices))
    return acidic_indices


def get_acidic_protons(atoms, atom_box, verbose=False):
    proton_indices = get_acidic_proton_indices(atoms[0], atom_box, verbose=verbose)
    protons = np.array(atoms[:, proton_indices]["pos"], order="C")
    return protons


def select_atoms(xyzatom_traj, *atomnames):
    """Select atoms from a trajectory of dtype \"dtype_xyz\"."""
    selections = []
    frames = xyzatom_traj.shape[0]
    for atomname in atomnames:
        if type(atomname) is str:
            atomname = atomname.encode()
        traj = xyzatom_traj[xyzatom_traj["name"] == atomname]["pos"]
        atomnumber = xyzatom_traj[0][xyzatom_traj[0]["name"] == atomname].size
        selections.append(np.array(traj.reshape((frames, atomnumber, 3)), order="C"))
    return selections


def map_indices(frame, atomname):
    """Returns indices of atomtype in original trajectory"""
    indices = []
    for i, atom in enumerate(frame):
        if atom["name"] == atomname:
            indices.append(i)
    return indices


def numpy_print(atoms, names=None, outfile=None):
    format_string = "{:4} {:20.10f} {:20.10f} {:20.10f}"
    if names is not None:
        atoms = atoms[atoms["name"] == name]
        print(sum([len(atoms[atoms["name"] == name]) for name in names]), file=outfile)
    else:
        print(len(atoms), file=outfile)
    if outfile is None:
        outfile = sys.stdout
    print("", file=outfile)
    for x in atoms:
        print(format_string.format("H" if x["name"] == "AH" else x["name"].decode("utf-8"),
                                   x["pos"][0], x["pos"][1], x["pos"][2]), file=outfile)


def print_npz(*args):
    try:
        npz_file = np.load(sys.argv[1])
    except:
        print("Usage:", sys.argv[0], "<npz filename>")
    for frame in npz_file["trajectory"]:
        numpy_print(frame)
    npz_file.close()


def remove_center_of_mass_movement(npa_traj):
    if npa_traj.shape[1] == 1:
        logger.info("Single atom trajectory. Will skip reduction of center of mass movement.")
        return None
    for name in npa_traj[0]["name"]:
        if name.astype(str) not in atom_masses:
            raise NameError("No atom mass specified for element {}".format(name))
    mass_array = np.array([atom_masses[name] for name in npa_traj[0]["name"].astype(str)])[None, :, None]
    center_of_mass = (mass_array * npa_traj["pos"]).sum(axis=1)[:, None, :] / mass_array.sum()
    npa_traj["pos"] -= center_of_mass


def print_center_of_mass(npa_traj):
    mass_array = np.array([atom_masses[name] for name in npa_traj[0]["name"]])[None, :, None]
    center_of_mass = (mass_array * npa_traj["pos"]).sum(axis=1)[:, None, :] / mass_array.sum()
    for i, com in enumerate(center_of_mass):
        print("Frame {:6d}".format(i), com)


def print_center_of_mass_commandline(*args):
    trajectory = np.load(sys.argv[1])["trajectory"]
    print_center_of_mass(trajectory)


class NeighborTopology:
    """Keeps track of the connections between atoms.
    Given a cutoff distance, for each atom the atoms within this
    distance will be determined."""
    def __init__(self, trajectory: Iterator[np.ndarray], cutoff: float, atombox: AtomBox,
                 buffer: float = 3.0) -> None:
        """
        Parameters
        ----------
        trajectory
        cutoff
        atombox
        buffer
        """
        self.trajectory = trajectory
        self.cutoff = cutoff
        self.atombox = atombox

    def get_topology_bruteforce(self, frame):
        """Determine the distance for each atom pair.
        If it is below the cutoff parameter, add it to the list
        of connections."""
        topology_matrix = lil_matrix((frame.shape[0], frame.shape[0]), dtype=float)
        for i, atom1 in enumerate(frame):
            for j, atom2 in enumerate(frame):
                if i != j:
                    dist = self.atombox.length(atom1, atom2)
                    if dist <= self.cutoff:
                        topology_matrix[i, j] = dist
        tocoo = topology_matrix.tocoo()
        return tocoo.row, tocoo.col, tocoo.data

    def topology_bruteforce_generator(self):
        for frame in self.trajectory:
            yield self.get_topology_bruteforce(frame)

    def get_topology_verlet_list(self):
        """Keep track of the two maximum atom displacements.
        As soon as their sum is larger than the buffer region, update
        the neighbor topology."""
        last_frame = None
        atombox = self.atombox
        logger.debug("start verlet list")
        for frame in self.trajectory:
            logger.debug(frame)
            if last_frame is None:
                yield self.get_topology_bruteforce(frame)
            else:
                dr = atombox.length(last_frame, frame)
                logger.debug("dr = %s", dr)
                yield 1, 2, 3

            last_frame = frame

