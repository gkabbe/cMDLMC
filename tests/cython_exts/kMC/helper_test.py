#!/usr/bin/python

import sys
import numpy as np
import time
import unittest

sys.path.append("/home/kabbe/PhD/pythontools/cython_exts/kMC")
import kMC_helper
from kMC.MDMC import create_celllists_simple_z, neighborlists

def print_frame(atoms):
		print atoms.shape[0]
		print ""
		for j in xrange(atoms.shape[0]):
			print "{:>2} {: 20.10f} {: 20.10f} {: 20.10f}".format("O", *atoms[j])

def extend_simulationbox(box_multiplier, Opos, pbc, oxygennumber):
	Oview = Opos.view()
	Oview.shape = (box_multiplier[0], box_multiplier[1], box_multiplier[2], oxygennumber, 3)
	for x in xrange(box_multiplier[0]):
		for y in xrange(box_multiplier[1]):
			for z in xrange(box_multiplier[2]):
				for i in xrange(oxygennumber):
					if x+y+z != 0:
						Oview[x, y, z, i, :] = Oview[0,0,0,i] + pbc*[x,y,z]

Os = np.load(sys.argv[1])
r_cut = float(sys.argv[2])

Olarge = np.zeros((5*144, 3), np.float32)
Olarge[:144, :] = Os[0]


parameters_fs = np.array([0.4*0.06,2.363,0.035])
pbc = np.array([29.122, 25.354, 12.363])
pbc_new = np.array([29.122, 25.354, 5*12.363])


# atomcount, cellindices, cellshape = create_celllists_simple_z(Olarge, pbc_new, r_cut, verbose=True)

# nbs = neighborlists(Olarge, pbc, r_cut)
# for k in nbs.keys():
# 	print k, nbs[k]
# print "Time Python", time.time() - start_time





# print_frame(Olarge)

helper = kMC_helper.Helper()

# sys.exit()
kMC_helper.extend_simulationbox_z(5, Olarge, pbc[2], 144)
# nl=helper.get_neighbors(Olarge, pbc, r_cut)
helper.determine_neighbors(Os[0], pbc, r_cut)

tm_old = np.zeros((144, 144))
tm_new = np.zeros((144, 144))

# start_time = time.time()
# for i in xrange(75000):
# 	if i%100 == 0:
# 		# helper.get_neighbors(Olarge, pbc, r_cut)
# 		print float(i)/(time.time()-start_time)
# 		print "\r",

# 	helper.calculate_transitionsmatrix_old(Os[i], tm_old, parameters_fs, pbc, r_cut)

# np.save("tm_old", tm_old)

start_time = time.time()
for i in xrange(75000):
	if i%1000 == 0:
		helper.determine_neighbors(Olarge, pbc, r_cut)
		print float(i)/(time.time()-start_time)
		print "\r",

	helper.calculate_transitionsmatrix_new(Os[i], tm_new, parameters_fs, pbc, r_cut)

np.save("tm_new", tm_new)

	# Olarge[:144, :] = Os[i]
	# # extend_simulationbox((1,1,5), Olarge, pbc, 144)
	# kMC_helper.extend_simulationbox_z(5, Olarge, pbc[2], 144)
	# # print_frame(Olarge)
	# start_old, dest_old, problist_old = helper.calculate_transitions_old(Olarge, parameters_fs, pbc, r_cut)
	# start_new, dest_new, problist_new = helper.calculate_transitions_new(Olarge, parameters_fs, pbc, r_cut)

	# for i, prob_old in enumerate(problist_old):
	# 	if prob_old - problist_new[i] > 1e-8:
	# 		print "ohoh"
	# 		print "frame", i
	# 		print prob_old, problist_new[i]
	# 		sys.exit()



# print ""
# print time.time() - start_time

start, dest, prob = helper.return_transitions()

# for i in xrange(len(start)):
# 	print start[i], dest[i], prob[i]

# print len(start)

# print type(start)
