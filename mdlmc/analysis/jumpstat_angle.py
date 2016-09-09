#!/usr/bin/env python3 -u

import sys
import numpy as np
import time
import ipdb
import argparse
import pickle
import os
import re
import inspect
# import pyximport; pyximport.install()

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(script_path, "../cython/atoms"))
import numpyatom as npa

sys.path.append(os.path.join(script_path, "../cython/LMC"))
import jumpstat_helper as jsh

from atoms import atomclass
from atoms.atomclass import hbond
from IO import BinDump

#class OHO:
#	def __init__(self, covO, freeO, H):
#		self.covO = covO
#		self.freeO = freeO
#		self.H = H
#		
#	def __eq__(self, other):
#		return isinstance(other, self.__class__) and self.__dict__ == other.__dict__
#
#def get_neighbors(Os, r_cut, pbc):
#	neighbors = [[] for i in xrange(Os.shape[0])]
#	for i, O1 in enumerate(Os):
#		for j, O2 in enumerate(Os):
#			if npa.length(O1, O2, pbc) < r_cut and i != j:
#				neighbors[i].append(j)
#
#	return neighbors
#
#def getOHO(Hs, Os, covs, occ_counter, OHOhisto, dmin, dmax, bins, pbc, neighbors):
#	OHOs=[[] for i in xrange(bins)]
#	occupied=[]
#	for i in xrange(Hs.shape[0]):
#		occupied.append(covs[i])
#		
#	for (Hind, Oind) in enumerate(occupied):
#		# for j in xrange(Os.shape[0]):
#		for j in neighbors[Oind]:
#			if not j in occupied and i/3 != j/3:
#				#~ pdb.set_trace()
#				#convert numpypy.uint8 back to int to make indexing work again....
#				difflen = npa.length(Os[int(Oind)], Os[j], pbc)
#				if dmax > difflen >= dmin:
#					OHOs[int(float(difflen-dmin)/(dmax-dmin)*bins)].append(OHO(Oind, j, Hind))
#					
#	for (i, item) in enumerate(OHOs):
#		if item != []:
#			occ_counter[i] += 1
#			OHOhisto[i] += len(item)
#					
#	return OHOs
#	
#def jump_histo(filename, dmin, dmax, bins, progress, pbc, frames, Os, Hs, covevo):
#	"""Counts proton jumps at different distances"""
#	start_time = time.time()
#
#	jumpcounter = np.zeros(bins, int)
#
#	for frame in xrange(Os.shape[0]-1):
#		neighborchange = covevo[frame]!=covevo[frame+1]
#		if neighborchange.any():
#			jump_protons = neighborchange.nonzero()[0]
#			for i in jump_protons:
#				# pdb.set_trace()
#				O_before = covevo[frame, i]
#				O_after = covevo[frame+1, i]
#				O_dist = npa.length(Os[frame, O_after], Os[frame, O_before], pbc)
#				jumpcounter[(O_dist-dmin)/(dmax-dmin)*bins] += 1
#
#	print "#Proton jump histogram:"
#	for i in xrange(jumpcounter.size):
#		print "{:10} {:10}".format(dmin+(dmax-dmin)/bins*(.5+i), jumpcounter[i])
#
#	print "#Jumps total: {:}".format(jumpcounter.sum())
#
#def OHOcompare(Hs, Os, OHOs, covs, evopart):
#	for (i, OHObin) in enumerate(OHOs):
#		#~ evopart[i] = len(OHObin)
#		for OHOobj in OHObin:
#			#~ pdb.set_trace()
#			if covs[OHOobj.H] == OHOobj.freeO:
#				evopart[i] += 1./len(OHObin)
#
#def OHOcompare2(Hs, Os, OHOs, covs, evopart):
#	for (i, OHObin) in enumerate(OHOs):
#		summe=0
#		#~ evopart[i] = len(OHObin)
#		for OHOobj in OHObin:
#			#~ pdb.set_trace()
#			if covs[OHOobj.H] == OHOobj.freeO:
#				summe += 1./len(OHObin)		
#			evopart[i].append(summe)
#
#def jump_probs(filename, dmin, dmax, bins, progress, pbc, frames, Os, Hs, covevo, neighbors):	
#
#	start_time = time.time()
#	
#	# transferrate_coll = np.zeros(bins, float)
#	transferrate_coll = [[] for i in xrange(bins)]
#	occurrencecounter = np.zeros(bins, int)
#	OHOhisto = np.zeros(bins, int)
#	#~ pdb.set_trace	
#	#~ for frame in xrange(0, Oarr.shape[0], step):
#	for frame in xrange(Os.shape[0]-1):
#		OHOs = getOHO(Hs[frame], Os[frame], covevo[frame], occurrencecounter, OHOhisto, dmin, dmax, bins, pbc, neighbors)
#		try:
#			#~ OHOcompare(Hs[frame+1], Os[frame+1], OHOs, covevo[frame+1], transferrate)
#			OHOcompare2(Hs[frame+1], Os[frame+1], OHOs, covevo[frame+1], transferrate_coll)
#		except IndexError:
#			pdb.set_trace()
#		if progress == True and frame%100 == 0:
#			# print >> sys.stderr, frame, "/", Os.shape[0]
#			print >> sys.stderr, "{:>6}/{:<6} ({:.0f} fps)".format(frame, Os.shape[0], float(frame)/(time.time()-start_time)),
#			print >> sys.stderr, "\r",
#	print ""
#			
#	
#	#~ for i in range(transferrate.shape[0]):
#		#~ if occurrencecounter[i] > 0:
#			#~ print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, transferrate[i]/occurrencecounter[i], OHOhisto[i], occurrencecounter[i]
#		#~ else:
#			#~ print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, 0, occurrencecounter[i], OHOhisto[i], occurrencecounter[i]
#	for i in range(transferrate.shape[0]):
#		if occurrencecounter[i] > 0:
#			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, np.array(transferrate_coll[i]).mean(), np.array(transferrate_coll[i]).var(), OHOhisto[i], occurrencecounter[i]
#		else:
#			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, 0, occurrencecounter[i], OHOhisto[i], occurrencecounter[i]
#
#def jump_probs2(Os, Hs, covevo, pbc, dmin, dmax, bins, verbose=False):
#	# pdb.set_trace()
#	possible_jumpers = np.zeros(bins, int)
#	actual_jumpers = np.zeros(bins, int)
#	probhisto = [[] for i in xrange(bins)]
#
#
#	for i in xrange(Os.shape[0]-1):
#		print "i:",i,
#		possible_jumpers.fill(0)
#		actual_jumpers.fill(0)
#		for O_i in xrange(Os[i].shape[0]):
#			print "O_i:", O_i,
#			for H_ind, O_j in enumerate(covevo[i]):
#				if O_i != O_j and not O_i in covevo:
#					distance = npa.length(Os[i, O_i], Os[i, O_j], pbc)
#					if distance < dmax:
#						binind = int((distance-dmin)/(dmax-dmin)*bins)
#						possible_jumpers[binind] += 1
#						if covevo[i+1, H_ind] != covevo[i, H_ind]:
#							actual_jumpers[binind] += 1
#
#		for j in xrange(possible_jumpers.size):
#			if possible_jumpers[j] != 0:
#				probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
#		print "jojo"
#		if verbose == True:
#			if i%1 == 0:
#				print "#", i,"\r",
#
#
#	print "# Oxygen distance, jump probability, jump probability standard deviation" 
#	for i in xrange(bins):
#		if len(probhisto[i]) > 0:
#			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())
#
def determine_oxy_neighbor_pairs(O_frame, neighbor_frame, pbc):
	neighbors = np.zeros(O_frame.shape[0], int)
	for i in range(O_frame.shape[0]):
		nb_index = npa.next_neighbor(O_frame[i], neighbor_frame, pbc)[0]
		neighbors[i] = nb_index
	return neighbors

#def jump_probs_POPOangle(Os, Ps, Ps_list, Hs, covevo, pbc, angle_min_rad, angle_max_rad, dmax, bins, verbose=False):
#	# pdb.set_trace()
#	possible_jumpers = np.zeros(bins, int)
#	actual_jumpers = np.zeros(bins, int)
#	probhisto = [[] for i in xrange(bins)]
#
#	for i in xrange(Os.shape[0]-1):
#		#~ print "i:",i,
#		possible_jumpers.fill(0)
#		actual_jumpers.fill(0)
#		for O_i in xrange(Os[i].shape[0]):
#			#~ print "O_i:", O_i,
#			for H_ind, O_j in enumerate(covevo[i]):
#				if O_i != O_j and not O_i in covevo:
#					distance = npa.length(Os[i, O_i], Os[i, O_j], pbc)
#					if distance < dmax:
#						angle = npa.angle_4_rad(Ps[i, Ps_list[O_i]], Os[i, O_i], Ps[i, Ps_list[O_j]], Os[i, O_j], pbc)
#						#TODO: determine POPO angle
#						binind = int((angle-angle_min_rad)/(angle_max_rad-angle_min_rad)*bins)
#						possible_jumpers[binind] += 1
#
#						if covevo[i+1, H_ind] != covevo[i, H_ind]:
#							actual_jumpers[binind] += 1
#
#		for j in xrange(possible_jumpers.size):
#			if possible_jumpers[j] != 0:
#				probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
#		if verbose == True:
#			if i%1 == 0:
#				print "#", i,"\r",
#
#
#	print "# Oxygen distance, jump probability, jump probability standard deviation" 
#	for i in xrange(bins):
#		if len(probhisto[i]) > 0:
#			print angle_min_rad + (angle_max_rad-angle_min_rad)/bins/2 + i*(angle_max_rad-angle_min_rad)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

def main(*args):

	parser=argparse.ArgumentParser(description="Jumpstats for angle dependency. If oxygens are not bound to phosphorus, specify neighbor via argument --oxygen_neighbor")
	parser.add_argument("filename", help="trajectory")
	parser.add_argument("pbc", type=float, nargs=3, help="Periodic boundaries")
	parser.add_argument("--oxygen_neighbor", "-o",default="P", help = "Atom to which the oxygens are bound")
	parser.add_argument("--angle_min", type=float, default=0., help = "Minimal angle value in Histogram (degrees)")
	parser.add_argument("--angle_max", type=float, default=180., help = "Maximal angle value in Histogram (degrees)")
	parser.add_argument("--bins", type=int, default=100, help = "Bins of histogram")
	parser.add_argument("--dmax", type=float, default=3., help = "Maximal OO distance, for which the angle is determined")
	parser.add_argument("--verbose", "-v", action = "store_true", default="False", help="Verbosity")
	args = parser.parse_args()

	pbc= np.array(args.pbc)

	if args.verbose == True:
		print("#PBC used:", pbc)

	trajectory = BinDump.npload_atoms(args.filename, create_if_not_existing=True, verbose=args.verbose)
	BinDump.mark_acidic_protons(trajectory, pbc, verbose=args.verbose)
	atoms = npa.select_atoms(trajectory, "O", args.oxygen_neighbor, "AH")
	Os = atoms["O"]
	Hs = atoms["AH"]
	oxy_nb = atoms[args.oxygen_neighbor]

	oxy_nb_list = determine_oxy_neighbor_pairs(Os[0], oxy_nb[0], pbc)

	covevo_filename = re.sub("\..{3}$", "", args.filename)+"_covevo.npy"
	if not os.path.exists(covevo_filename):
		BinDump.npsave_covevo(covevo_filename, Os, Hs, pbc, verbose=args.verbose)
	covevo = np.load(covevo_filename)
		
	angle_min = np.radians(args.angle_min)
	angle_max = np.radians(args.angle_max)

	jsh.jump_probs_angle_POO(Os, oxy_nb, oxy_nb_list, Hs, covevo, pbc, angle_min, angle_max, args.dmax, args.bins, verbose=args.verbose)


if __name__ == "__main__":
	main()
