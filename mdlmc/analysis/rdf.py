import numpy as np
import argparse
import ipdb
import warnings
import matplotlib.pylab as plt
import time

from mdlmc.IO import BinDump
import mdlmc.atoms.numpyatom as npa


def distance(a1, a2, pbc):
    diff = a2-a1
    while (diff > pbc / 2).any():
        diff = np.where(diff > pbc/2, diff-pbc, diff)
    while (diff < -pbc / 2).any():
        diff = np.where(diff < -pbc/2, diff+pbc, diff)
    return np.sqrt(np.sum(diff**2, axis=-1))


def rdf(traj1, traj2, pbc, histo_kwargs):
    histogram = np.zeros(histo_kwargs["bins"], dtype=int)
    
    frame1, frame2 = traj1[0], traj2[0]
    
    neighbor_lists = [np.where(distance(a, frame2, pbc) <= histo_kwargs["range"][1] + 2) for a in frame1]
    # ipdb.set_trace()
    start_time = time.time()
    diffs = []
    for i, (frame1, frame2) in enumerate(zip(traj1, traj2)):
        for j, atom in enumerate(frame1):
            diffs += list(distance(atom, frame2[neighbor_lists[j]], pbc))
        if i % 1000 == 0:
            print(float(i)/(time.time()-start_time), "frames/sec")
            histo, edges = np.histogram(diffs, **histo_kwargs)
            histogram += histo
            diffs = []
    dists = (edges[:-1] + edges[1:])/2
    
    print("# Total time:", time.time() - start_time)
    
    return histogram, dists, edges


def main(*args):
    parser=argparse.ArgumentParser(description="Calculates RDF")
    parser.add_argument("file", help="trajectory")
    parser.add_argument("pbc", nargs=3, type=float, help="Periodic boundaries")
    parser.add_argument("minval", type=float, help="minimal value")
    parser.add_argument("maxval", type=float, help="maximal value")
    parser.add_argument("bins", type=int, help="number of bins")
    parser.add_argument("--cut", type=int, help="Cut trajectory after frame")
    parser.add_argument("-e", "--elements", nargs="+", default=["O"], help="which elements (use AH for acidic protons)")
    args = parser.parse_args()
    
    histo_kwargs = {
                    "range": (args.minval, args.maxval),
                    "bins": args.bins      
                    }

    if len(args.elements) > 2:
        warnings.warn("Received more than two elements. Will just skip elements after the first two")

    traj = BinDump.npload_atoms(args.file)[:args.cut]
    pbc = np.array(args.pbc)
    
    if "AH" in args.elements:
        BinDump.mark_acidic_protons(traj, pbc)
    
    selection = npa.select_atoms(traj, *args.elements)
    
    if len(args.elements) >= 2:
        elem1, elem2 = selection[:2]
    else:
        elem1 = selection
        elem2 = elem1
        
    histo, dists, edges = rdf(elem1, elem2, pbc, histo_kwargs)
    dr = dists[1] - dists[0]
    N = elem2.shape[1]
    V = pbc[0] * pbc[1] * pbc[2]
    rho = N/V
    
    print("# Number of particles:", N)
    print("# Volume:", V)
    print("# Rho = N/V =", rho)
    print(traj.shape)
    
    histo_norm = np.asfarray(histo) / (4./3*np.pi*rho*(edges[1:]**3-edges[:-1]**3)) / elem2.shape[1] / traj.shape[0]
    # histo_norm2 = np.asfarray(histo) / (4./3*np.pi*rho*dists**2*dr) / elem2.shape[1] / traj.shape[0]
    # plt.plot(dists, histo_norm, dists, histo_norm2)
    # plt.show()
    
    output = np.hstack([dists[:, None], histo_norm[:, None]])
    np.savetxt("rdf_{}".format(args.cut), output)
