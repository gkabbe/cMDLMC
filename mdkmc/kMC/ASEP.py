#!/usr/bin/python

import numpy as np
import sys
import time
import matplotlib.pylab as plt

sys.path.append("/home/kabbe/PhD/pythontools/cython/ASEP")
import tasep

def insdel_rate(x, lambd):
	return 0.5*(1+np.cos(x/lambd*np.pi))


sweeps = 10**5
latticesize = 120

#node i is connected to node i+1 and vice versa
transitionmatrix = (np.eye(latticesize, k=1) + np.eye(latticesize, k=-1))

#jumps from i to 0 (source) not possible -> tm[:,0] = 0
transitionmatrix[:, 0] = 0
#jumps from latticesize-1 (drain) to i not possible -> tm[latticesize-1,:] = 0
transitionmatrix[latticesize-1, :] = 0

#set insertion (tm[0,:]) and deletion rate (tm[:,latticesize-1])
transitionmatrix[0, :] = 0
transitionmatrix[0, 1:21] = insdel_rate(np.arange(20, dtype=float), 20)
transitionmatrix[:, -1] = 0
transitionmatrix[-2:-22:-1, -1] = insdel_rate(np.arange(20, dtype=float), 20)

#bottleneck
transitionmatrix[99,100] = 0.3

lattice = np.zeros(transitionmatrix.shape[0], np.uint8)

print("prerun")
tasep.asep_matrix(lattice, transitionmatrix, sweeps/10, 1)
print("run")
counter, steps = tasep.asep_matrix(lattice, transitionmatrix, sweeps, 2)


density = counter / float(steps)

plt.plot(density)
plt.show()