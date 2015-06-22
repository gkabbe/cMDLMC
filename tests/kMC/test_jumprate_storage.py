import numpy as np

from mdkmc.IO import BinDump
from mdkmc.cython_exts.kMC.jumprate_storage import JumprateStorage as JS


trajectory = BinDump.npload_atoms("/home/kabbe/trajectories/400Kbeginning.xyz", atomnames_list=["O"])
pbc = np.array([29.122, 25.354, 12.363])
jumprate_parameter_dict = {"a": 0.5, "b": 2.3, "c": 0.1}

trajectory = np.array(trajectory["pos"], order="C")
print trajectory.shape

js = JS(trajectory, pbc, jumprate_parameter_dict)


for i in range(100):
    js.get_values(i)