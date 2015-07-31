#!/usr/bin/python -u
import numpy as np
import argparse
import sys
import ipdb
import re
import os

from kMC import MDMC
from IO import BinDump
sys.path.append("/home/kabbe/PhD/pythontools/cython/atoms")
import numpyatom as npa

def get_nearest_neighbors_traj(Hs, Os, pbc, verbose=False):
	# ipdb.set_trace()
	nearest_neighbors = np.zeros((Hs.shape[0], Hs.shape[1]), int)
	for i in xrange(Hs.shape[0]):
		for j in xrange(Hs.shape[1]):
			nearest_neighbors[i,j] = npa.nextNeighbor(Hs[i,j], Os[i], pbc)[0]
		if verbose == True and i % 1000 == 0:
			print "#{: 6d}/{: 6d}".format(i, Hs.shape[0]), "\r",
	if verbose == True:
		print ""
	return nearest_neighbors

def jumpmatrix_MD(covevo, O_number, verbose=False):
	counter = 0
	jumpmat = np.zeros((O_number, O_number), int)
	
	if verbose == True:
		print "#evaluating jumps"

	for i in xrange(1, covevo.shape[0]):
		jump_indices = np.where(covevo[i] != covevo[i-1])[0]
		if jump_indices.size > 0:
			counter += 1
			print "{} jumps".format(counter), "\r",
			for ind in jump_indices:
				jumpmat[covevo[i-1, ind], covevo[i, ind]] += 1
	print ""

	return jumpmat

def jumpmatrix_kMC(filename, pbc, temp, proton_number, time_step, sweeps, equilibration_sweeps, mode, verbose=False):
	mdmc = MDMC.MDMC(filename=filename, pbc=pbc, temp=temp, proton_number=proton_number)
	if mode == "dist":
		jumpmat = mdmc.kmc_run(output=sys.stdout, sweeps=sweeps, equilibration_sweeps=equilibration_sweeps, MD_timestep_fs=time_step, skip_frames=0, print_freq=75, reset_freq=25000, dump_trajectory=False, box_multiplier=(1,1,1), neighbor_freq=10, verbose=verbose)
	elif mode == "angle":
		jumpmat = mdmc.ASEP_angle(output=sys.stdout, sweeps=sweeps, equilibration_sweeps=equilibration_sweeps, MD_timestep_fs=time_step, skip_frames=0, print_freq=75, reset_freq=25000, dump_trajectory=False, box_multiplier=(1,1,1), neighbor_freq=10, verbose=verbose)
	else:
		raise NotImplementedError("Unknown mode {}".format(mode))
	return jumpmat

def compare_matrices(mdmat, kmcmat):
	mdmat_normed = mdmat/float(mdmat.sum())
	kmcmat_normed = kmcmat/float(kmcmat.sum())

	return kmcmat_normed - mdmat_normed

if __name__ == "__main__":
	parser=argparse.ArgumentParser(description="Count proton jumps in MD and kMC and compare jump matrices")
	parser.add_argument("filename", help="Trajectory")
	parser.add_argument("pbc", type=float, nargs=3, help="Periodic boundaries")
	parser.add_argument("--mode", "-m", choices=["dist", "angle"], default="dist", help="Choose between distance criterion and distance+angle criterion")
	parser.add_argument("timestep", type=float, help="Timestep used in MD")
	parser.add_argument("--temp", type=float, help="Temperature")
	parser.add_argument("--verbose", "-v", action = "store_true", default="False", help="Verbosity")
	args = parser.parse_args()

	if args.temp != None:
		temp = args.temp
	else:
		temp = int(re.findall("\d+", args.filename)[0])
	
	print "#using temperature {}".format(temp)

	kmcmat_filename = args.filename[:-4]+"_jumpMD.npy"
	mdmat_filename = args.filename[:-4]+"_jumpkMC.npy"

	if os.path.exists(kmcmat_filename) and os.path.exists(mdmat_filename):
		answer = raw_input("{} and {} already exist. Load? (y,n)\n".format(kmcmat_filename, mdmat_filename))
		if answer.upper() == "Y":
			jumpmat_MD = np.load(mdmat_filename)
			jumpmat_kMC = np.load(kmcmat_filename)

	else:
		pbc = np.array(args.pbc)
		traj = BinDump.npload_atoms(args.filename, create_if_not_existing=True, verbose=args.verbose)
		BinDump.mark_acidic_protons(traj, pbc, args.verbose)	
		Os = traj[traj["name"]=="O"]
		Hs = traj[traj["name"]=="AH"]
		Os = np.array(Os.reshape((traj.shape[0], Os.shape[0]/traj.shape[0]))["pos"], order="C")
		Hs = np.array(Hs.reshape((traj.shape[0], Hs.shape[0]/traj.shape[0]))["pos"], order="C")

		proton_number = Hs.shape[1]

		if args.verbose == True:
			print "#Determined proton number:", proton_number

		covevo_filename = args.filename[:-4]+"_covevo.npy"
		if os.path.exists(covevo_filename):
			print "#found covevo in {}".format(covevo_filename)
			covevo = np.load(covevo_filename)
		else:
			covevo = BinDump.npsave_covevo(covevo_filename, Os, Hs, pbc, args.verbose)

		jumpmat_MD = jumpmatrix_MD(covevo, Os.shape[1], args.verbose)
		jumpmat_kMC = jumpmatrix_kMC(filename=args.filename, pbc=pbc, temp=temp, proton_number=proton_number, time_step=args.timestep, sweeps=Os.shape[0], equilibration_sweeps=2500000, verbose=args.verbose, mode=args.mode)

		np.save(args.filename[:-4]+"_jumpMD"+str(temp), jumpmat_MD)
		np.save(args.filename[:-4]+"_jumpkMC"+str(temp), jumpmat_kMC)

		# ipdb.set_trace()

	compmat = compare_matrices(jumpmat_MD, jumpmat_kMC)

	np.savetxt(args.filename[:-4]+"_compmat", compmat)
