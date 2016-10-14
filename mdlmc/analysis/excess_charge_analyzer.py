import argparse

from mdlmc.IO import xyz_parser


def excess_charge_collective_variable(oxygen_pos, proton_pos):
    return proton_pos.sum(axis=0) - 2 * oxygen_pos.sum(axis=0)


def main(*args):
    parser = argparse.ArgumentParser(
        description="Determine the excess charge movement in a water box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trajectory", help="Trajectory in npz format")
    parser.add_argument("--visualize", action="store_true", help="Visualize excess charge position")
    parser.add_argument("pbc", nargs=3, help="periodic boundary conditions")
    args = parser.parse_args()

    oxygens, protons = xyz_parser.load_trajectory_from_npz(args.trajectory, "O", "H")
    if args.visualize:
        atoms = xyz_parser.load_atoms(args.trajectory)

    for i, (oxygen_frame, proton_frame) in enumerate(zip(oxygens, protons)):
        # Like in the PVPA paper, create a collective variable that tracks the excess charge motion
        excess_charge_colvar = excess_charge_collective_variable(oxygen_frame, proton_frame)
        if args.visualize:
            print(len(atoms[i]) + 1)
            print()
            for atom in atoms[i]:
                print(atom["name"], " ".join(map(str, atom["pos"])))
        print("S", " ".join(map(str, excess_charge_colvar)), flush=True)
