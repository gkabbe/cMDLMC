import argparse

from mdlmc.IO import xyz_parser


def main(*args):
    parser = argparse.ArgumentParser(
        description="cMD/LMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trajectory", help="Trajectory in npz format")
    parser.add_argument("pbc", nargs=3, help="periodic boundary conditions")
    args = parser.parse_args()

    oxygens, protons = xyz_parser.load_trajectory_from_npz(args.trajectory, "O", "H")

    for oxygen_frame, proton_frame in zip(oxygens, protons):
        # Like in the PVPA paper, create a collective variable that tracks the excess charge motion
        excess_charge_colvar = proton_frame.sum(axis=0) - 2 * oxygen_frame.sum(axis=0)
        print(" ".join(map(str, excess_charge_colvar)), flush=True)
