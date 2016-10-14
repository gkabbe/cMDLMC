import argparse

import numpy as np
from mdlmc.IO import xyz_parser
from mdlmc.cython_exts.LMC.PBCHelper import AtomBoxCubic, AtomBoxMonoclinic


def excess_charge_collective_variable(oxygen_pos, proton_pos):
    return proton_pos.sum(axis=0) - 2 * oxygen_pos.sum(axis=0)


def determine_hydronium_position(oxygen_pos, proton_pos, atombox):
    closest_oxygen_indices = np.zeros(proton_pos.shape[0], dtype=int)
    for proton_index, proton in enumerate(proton_pos):
        oxygen_index, _ = atombox.next_neighbor(proton, oxygen_pos)
        closest_oxygen_indices[proton_index] = oxygen_index
    for i in range(oxygen_pos.shape[0]):
        if (closest_oxygen_indices == i).sum() == 3:
            return oxygen_pos[i]
    else:
        raise RuntimeError("Could not determine excess charge position.")


def main(*args):
    """Determine the excess charge movement in a water box"""
    parser = argparse.ArgumentParser(
        description="Determine the excess charge movement in a water box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--visualize", action="store_true", help="Visualize excess charge position")
    parser.add_argument("trajectory", help="Trajectory in npz format")
    parser.add_argument("pbc", nargs=3, type=float, help="periodic boundary conditions")
    subparsers = parser.add_subparsers()
    parser_cv = subparsers.add_parser("cv", help="Track evolution of collective variable")
    parser_cv.set_defaults(func=track_collective_variable)
    parser_occupation = subparsers.add_parser("occupation", help="Track position of H3O+ ion")
    parser_occupation.set_defaults(func=track_hydronium_ion)
    args = parser.parse_args()
    args.func(args)


def track_collective_variable(args):
    pbc = np.array(args.pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_trajectory_from_npz(args.trajectory, "O", "H")
    excess_charge_start_position = determine_hydronium_position(oxygens[0], protons[0], atombox)
    excess_charge_colvar_0 = excess_charge_collective_variable(oxygens[0], protons[0])
    if args.visualize:
        atoms = xyz_parser.load_atoms(args.trajectory)

    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        # Like in the PVPA paper, create a collective variable that tracks the excess charge motion
        excess_charge_colvar = excess_charge_collective_variable(oxygen_frame, proton_frame) \
                               - excess_charge_colvar_0 + excess_charge_start_position
        if args.visualize:
            print(len(atoms[i]) + 1)
            print()
            for atom in atoms[i]:
                print(atom["name"], " ".join(map(str, atom["pos"])))
        print("S", " ".join(map(str, excess_charge_colvar)), flush=True)


def track_hydronium_ion(args):
    pbc = np.array(args.pbc)
    atombox = AtomBoxCubic(pbc)
    oxygens, protons = xyz_parser.load_trajectory_from_npz(args.trajectory, "O", "H")
    if args.visualize:
        atoms = xyz_parser.load_atoms(args.trajectory)
    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        hydronium_position = determine_hydronium_position(oxygen_frame, proton_frame, atombox)
        if args.visualize:
            print(len(atoms[i]) + 1)
            print()
            for atom in atoms[i]:
                print(atom["name"], " ".join(map(str, atom["pos"])))
        print("S", " ".join(map(str, hydronium_position)), flush=True)