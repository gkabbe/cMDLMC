from mdkmc.IO import BinDump
from mdkmc.atoms import numpyatom as npa

import sys
import ipdb


def main(*args):
    try:
        traj = BinDump.npload_atoms(sys.argv[1], verbose=True)    
    except IndexError:
        print "Usage: {} <filename.xyz>".format(sys.argv[0])
        
    oxygen_traj = npa.select_atoms(traj, "O")

    
