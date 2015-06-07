import numpy as np
import sys
import time
import cython

cimport numpy as cnp

from cython_exts.kMC.kMC_helper cimport fermi# float_or_double_array_1d, float_or_double
cimport cython

from cython_exts.atoms cimport numpyatom as cnpa

from libcpp.vector cimport vector

from libc.math cimport sqrt, exp

cdef extern from "math.h":
	double sqrt(double x) nogil
	double exp(double x) nogil
	double cos(double x) nogil
	double acos(double x) nogil

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int contains_ptr(cnp.int64_t x, cnp.int64_t * vec, int size):
	cdef int i
	for i in range(size):
		if x == vec[i]:
			return 1
	return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int contains(cnp.int64_t x, cnp.int64_t [:] vec):
	cdef int i
	for i in range(vec.shape[0]):
		if x == vec[i]:
			return 1
	return 0


# cdef int neighborchange(int [:] cov1, int [:] cov2):
# 	cdef int i

# 	for i in range(cov1.size):
# 		if cov1[i] == cov2[i]:
# 			return True
# 	return False

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# cdef double length(double [:] a1_pos, double [:] a2_pos, double [:] pbc):
cdef inline double length_ptr(double * a1_pos, double * a2_pos, double * pbc) nogil:
	cdef: 
		double *dist = [a2_pos[0] - a1_pos[0], a2_pos[1] - a1_pos[1], a2_pos[2] - a2_pos[2]]
		int i

	for i in range(3):
		dist[i] = a2_pos[i] - a1_pos[i]
		while dist[i] < -pbc[i]/2:
			dist[i] += pbc[i]
		while dist[i] > pbc[i]/2:
			dist[i] -= pbc[i]
	return sqrt(dist[0]*dist[0]+dist[1]*dist[1]+dist[2]*dist[2])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double length(double [:] a1_pos, double [:] a2_pos, double [:] pbc):
	cdef: 
		double *dist = [a2_pos[0] - a1_pos[0], a2_pos[1] - a1_pos[1], a2_pos[2] - a2_pos[2]]
		int i

	for i in range(3):
		dist[i] = a2_pos[i] - a1_pos[i]
		while dist[i] < -pbc[i]/2:
			dist[i] += pbc[i]
		while dist[i] > pbc[i]/2:
			dist[i] -= pbc[i]
	return sqrt(dist[0]*dist[0]+dist[1]*dist[1]+dist[2]*dist[2])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def jump_probs(double [:,:,::1] Os, double [:,:,::1] Hs, cnp.int64_t [:,::1] covevo, double [:] pbc, double dmin, double dmax, int bins, verbose=False, nonortho=False):
	# pdb.set_trace()
	cdef:
		cnp.int_t [:] possible_jumpers = np.zeros(bins, int)
		cnp.int_t [:] actual_jumpers = np.zeros(bins, int)
		int i, j, O_i, O_j, H_ind, binind
		double distance
		double * O_ptr = &Os[0,0,0]
		cnp.int64_t * cov_ptr = &covevo[0,0]
		vector[vector[double]] probhisto
		cnp.ndarray [cnp.double_t, ndim=2] h = np.array(np.array([[pbc[0], pbc[1], pbc[2]], [pbc[3], pbc[4], pbc[5]], [pbc[6], pbc[7], pbc[8]]]).T, order="C")
		cnp.ndarray [cnp.double_t, ndim=2] h_inv = np.array(np.linalg.inv(h), order="C")

	# probhisto = [[] for i in xrange(bins)]
	probhisto.resize(bins)

	start_time = time.time()

	for i in range(Os.shape[0]-1):
		possible_jumpers[:] = 0
		actual_jumpers[:] = 0
		for O_i in range(Os.shape[1]):
			for H_ind in range(covevo.shape[1]):
				O_j = covevo[i,H_ind]
				# if O_i != O_j and not contains(O_i, covevo[i]):
				if O_i != O_j and contains_ptr(O_i, cov_ptr+i*covevo.shape[1], covevo.shape[1]) != 1:
					if nonortho == False:
						distance = cnpa.length(Os[i, O_i], Os[i, O_j], pbc)
					else:
						distance = cnpa.length_nonortho_bruteforce_ptr(&Os[i, O_i, 0], &Os[i, O_j, 0], &h[0,0], &h_inv[0,0])
					# distance = length_ptr(O_ptr+i*Os.shape[1]*Os.shape[2]+O_i*Os.shape[2], O_ptr+i*Os.shape[1]*Os.shape[2]+O_j*Os.shape[2], &pbc[0])
					if distance < dmax and distance >=dmin:
						binind = int((distance-dmin)/(dmax-dmin)*bins)
						possible_jumpers[binind] += 1
						# if covevo[i+1, H_ind] != covevo[i, H_ind]:
						if covevo[i+1, H_ind] == O_i:
							actual_jumpers[binind] += 1

		for j in xrange(possible_jumpers.size):
			if possible_jumpers[j] != 0:
				# probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
				probhisto[j].push_back(float(actual_jumpers[j])/possible_jumpers[j])
		if verbose == True:
			if i%100 == 0:
				print "# {:} {:.2f} fps".format(i, i/(time.time()-start_time)),"\r",


	print "# Oxygen distance, jump probability, jump probability standard deviation" 
	for i in xrange(bins):
		if len(probhisto[i]) > 0:
			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def jump_probs_pbc_nonortho(double [:,:,::1] Os, double [:,:,::1] Hs, cnp.int64_t [:,::1] covevo, double [:,::1] pbc, double dmin, double dmax, int bins, verbose=False):
	# pdb.set_trace()
	cdef:

		cnp.int_t [:] possible_jumpers = np.zeros(bins, int)
		cnp.int_t [:] actual_jumpers = np.zeros(bins, int)
		int i, j, O_i, O_j, H_ind, binind
		double distance
		double * O_ptr = &Os[0,0,0]
		cnp.int64_t * cov_ptr = &covevo[0,0]
		vector[vector[double]] probhisto
		cnp.ndarray[cnp.double_t, ndim=2] h = pbc.T
		cnp.ndarray[cnp.double_t, ndim=2] h_inv = np.linalg.inv(pbc)
		cnp.ndarray[cnp.double_t, ndim=1] diffvec = np.zeros(3)

	# probhisto = [[] for i in xrange(bins)]
	probhisto.resize(bins)

	start_time = time.time()

	for i in range(Os.shape[0]-1):
		possible_jumpers[:] = 0
		actual_jumpers[:] = 0
		for O_i in range(Os.shape[1]):
			for H_ind in range(covevo.shape[1]):
				O_j = covevo[i,H_ind]
				# if O_i != O_j and not contains(O_i, covevo[i]):
				if O_i != O_j and contains_ptr(O_i, cov_ptr+i*covevo.shape[1], covevo.shape[1]) != 1:
					distance = cnpa.length_nonortho_bruteforce(Os[i, O_i], Os[i, O_j], h, h_inv)
#~ 					distance = sqrt(diffvec[0]*diffvec[0]+diffvec[1]*diffvec[1]+diffvec[2]*diffvec[2])
					if distance < dmax and distance >=dmin:
						binind = int((distance-dmin)/(dmax-dmin)*bins)
						possible_jumpers[binind] += 1
						# if covevo[i+1, H_ind] != covevo[i, H_ind]:
						if covevo[i+1, H_ind] == O_i:
							actual_jumpers[binind] += 1

		for j in xrange(possible_jumpers.size):
			if possible_jumpers[j] != 0:
				# probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
				probhisto[j].push_back(float(actual_jumpers[j])/possible_jumpers[j])
		if verbose == True:
			if i%100 == 0:
				print "# {:} {:.2f} fps".format(i, i/(time.time()-start_time)),"\r",

	print "# Oxygen distance, jump probability, jump probability standard deviation" 
	for i in xrange(bins):
		if len(probhisto[i]) > 0:
			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

#@cython.cdivision(True)
#@cython.wraparound(False)
#@cython.boundscheck(False)
#def jump_probs_angle_POPO(double [:,:,::1] Os, double [:,:,::1] Ps, cnp.int64_t [:] P_list, double [:,:,::1] Hs, cnp.int64_t [:,::1] covevo, double [:] pbc, double angle_min, double angle_max, double dmax, int bins, verbose=False):
#	# pdb.set_trace()
#	cdef:
#
#		cnp.int_t [:] possible_jumpers = np.zeros(bins, int)
#		cnp.int_t [:] actual_jumpers = np.zeros(bins, int)
#		int i, j, O_i, O_j, H_ind, binind
#		double distance, popo_angle
#		double * O_ptr = &Os[0,0,0]
#		cnp.int64_t * cov_ptr = &covevo[0,0]
#		vector[vector[double]] probhisto
#		
#	# probhisto = [[] for i in xrange(bins)]
#	probhisto.resize(bins)
#	
#	#nur zum test
#	for i in range(bins):
#		probhisto[i].push_back(0)
#
#	start_time = time.time()
#
#	for i in range(Os.shape[0]-1):
#		possible_jumpers[:] = 0
#		actual_jumpers[:] = 0
#		for O_i in range(Os.shape[1]):
#			for H_ind in range(covevo.shape[1]):
#				O_j = covevo[i,H_ind]
#				# if O_i != O_j and not contains(O_i, covevo[i]):
#				if O_i != O_j and contains_ptr(O_i, cov_ptr+i*covevo.shape[1], covevo.shape[1]) != 1:
#					distance = length(Os[i, O_i], Os[i, O_j], pbc)
#					# distance = length_ptr(O_ptr+i*Os.shape[1]*Os.shape[2]+O_i*Os.shape[2], O_ptr+i*Os.shape[1]*Os.shape[2]+O_j*Os.shape[2], &pbc[0])
#					if distance < dmax:
##~ 						popo_angle = angle(&Ps[i, P_list[O_i], 0], &Os[i, O_i, 0], &Ps[i, P_list[O_j], 0], &Os[i, O_j, 0], &pbc[0])
#						popo_angle = angle(Ps[i, P_list[O_i]], Os[i, O_i], Ps[i, P_list[O_j]], Os[i, O_j], pbc)
#						binind = int((popo_angle-angle_min)/(angle_max-angle_min)*bins)
#						possible_jumpers[binind] += 1
#						# if covevo[i+1, H_ind] != covevo[i, H_ind]:
#						if covevo[i+1, H_ind] == O_i:
#							actual_jumpers[binind] += 1
#
#		for j in xrange(possible_jumpers.size):
#			if possible_jumpers[j] != 0:
#				# probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
#				probhisto[j].push_back(float(actual_jumpers[j])/possible_jumpers[j])
#		if verbose == True:
#			if i%100 == 0:
#				print "# {:} {:.2f} fps".format(i, i/(time.time()-start_time)),"\r",
#
#
#	print "# Oxygen distance, jump probability, jump probability standard deviation" 
#	for i in xrange(bins):
#		if len(probhisto[i]) > 0:
#			print angle_min + (angle_max-angle_min)/bins/2 + i*(angle_max-angle_min)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def jump_probs_angle_POO(double [:,:,::1] Os, double [:,:,::1] Ps, cnp.int64_t [:] P_list, double [:,:,::1] Hs, cnp.int64_t [:,::1] covevo, double [:] pbc, double angle_min, double angle_max, double dmax, int bins, verbose=False):
	# pdb.set_trace()
	cdef:

		cnp.int_t [:] possible_jumpers = np.zeros(bins, int)
		cnp.int_t [:] actual_jumpers = np.zeros(bins, int)
		int i, j, O_i, O_j, H_ind, binind
		double distance, poo_angle, angl
		double * O_ptr = &Os[0,0,0]
		cnp.int64_t * cov_ptr = &covevo[0,0]
		vector[vector[double]] probhisto
		
	probhisto.resize(bins)
	
	#nur zum test
	for i in range(bins):
		probhisto[i].push_back(0)

	start_time = time.time()

	for i in range(Os.shape[0]-1):
		possible_jumpers[:] = 0
		actual_jumpers[:] = 0
		for O_i in range(Os.shape[1]):
			#check if O_i occupied
#~ 			if contains_ptr(O_i, cov_ptr+i*covevo.shape[1], covevo.shape[1]) != 1:
			if contains(O_i, covevo[i]) != 1:
				for H_ind in range(covevo.shape[1]):
					O_j = covevo[i,H_ind]
					# if O_i != O_j and not contains(O_i, covevo[i]):
					distance = length(Os[i, O_i], Os[i, O_j], pbc)
					# distance = length_ptr(O_ptr+i*Os.shape[1]*Os.shape[2]+O_i*Os.shape[2], O_ptr+i*Os.shape[1]*Os.shape[2]+O_j*Os.shape[2], &pbc[0])
					if distance < dmax:
						poo_angle = cnpa.angle(Os[i, O_j], Ps[i, P_list[O_j]], Os[i, O_j], Os[i, O_i], pbc)
						binind = int((poo_angle-angle_min)/(angle_max-angle_min)*bins)
						possible_jumpers[binind] += 1
						# if covevo[i+1, H_ind] != covevo[i, H_ind]:
						if covevo[i+1, H_ind] == O_i:
							actual_jumpers[binind] += 1

		for j in xrange(possible_jumpers.size):
			if possible_jumpers[j] != 0:
				# probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
				probhisto[j].push_back(float(actual_jumpers[j])/possible_jumpers[j])
		if verbose == True:
			if i%100 == 0:
				print "# {:} {:.2f} fps".format(i, i/(time.time()-start_time)),"\r",


	print "#Angle (radians), Angle (degrees), jump probability, jump probability standard deviation" 
	for i in xrange(bins):
		if len(probhisto[i]) > 0:
			angl = angle_min + (angle_max-angle_min)/bins/2 + i*(angle_max-angle_min)/bins
			print angl, np.degrees(angl), np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def jump_probs_neighborlist(double [:,:,::1] Os, double [:,:,::1] Hs, cnp.int64_t [:,::1] covevo, double [:] pbc, double dmin, double dmax, int bins, verbose=False):
	# pdb.set_trace()
	cdef:

		cnp.int_t [:] possible_jumpers = np.zeros(bins, int)
		cnp.int_t [:] actual_jumpers = np.zeros(bins, int)
		int i, j, O_i, O_j, H_ind, binind
		double distance
		double * O_ptr = &Os[0,0,0]
		cnp.int64_t * cov_ptr = &covevo[0,0]
		vector[vector[double]] probhisto

	# probhisto = [[] for i in xrange(bins)]
	probhisto.resize(bins)

	start_time = time.time()

	for i in range(Os.shape[0]-1):
		possible_jumpers[:] = 0
		actual_jumpers[:] = 0
		for O_i in range(Os.shape[1]):
			# for H_ind, O_j in enumerate(covevo[i]):
			for H_ind in range(covevo.shape[1]):
				O_j = covevo[i,H_ind]
				# if O_i != O_j and not contains(O_i, covevo[i]):
				if O_i != O_j and contains_ptr(O_i, cov_ptr+i*covevo.shape[1], covevo.shape[1]) != 1:
					# distance = length(Os[i, O_i], Os[i, O_j], pbc)
					distance = length_ptr(O_ptr+i*Os.shape[1]*Os.shape[2]+O_i*Os.shape[2], O_ptr+i*Os.shape[1]*Os.shape[2]+O_j*Os.shape[2], &pbc[0])
					if distance < dmax and distance >=dmin:
						binind = int((distance-dmin)/(dmax-dmin)*bins)
						possible_jumpers[binind] += 1
						if covevo[i+1, H_ind] != covevo[i, H_ind]:
							actual_jumpers[binind] += 1

		for j in xrange(possible_jumpers.size):
			if possible_jumpers[j] != 0:
				# probhisto[j].append(float(actual_jumpers[j])/possible_jumpers[j])
				probhisto[j].push_back(float(actual_jumpers[j])/possible_jumpers[j])
		if verbose == True:
			if i%100 == 0:
				print "# {:} {:.2f} fps".format(i, i/(time.time()-start_time)),"\r",


	print "# Oxygen distance, jump probability, jump probability standard deviation" 
	for i in xrange(bins):
		if len(probhisto[i]) > 0:
			print dmin + (dmax-dmin)/bins/2 + i*(dmax-dmin)/bins, np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var())

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int neighborchange(cnp.int64_t [:] covevo1, cnp.int64_t [:] covevo2):
#does not detect more than one jump between two frames. 
	cdef: 
		int i
	for i in range(covevo1.shape[0]):
		if covevo1[i] != covevo2[i]:
			return i
	return -1
	
cdef int neighborchange_ptr(cnp.int64_t *covevo1, cnp.int64_t *covevo2, int covsize):
#does not detect more than one jump between two frames. 
	cdef: 
		int i
	for i in range(covsize):
		if covevo1[i] != covevo2[i]:
			return i
	return -1
		

#def evaluate_model(double a, double b, double c, double [:, :, ::1] Os, double [:, :, ::1] Hs, cnp.int64_t [:, ::1] covevo, double [::1] pbc, verbose=False):
#	cdef: 
#		double prob = 1
#		double jumpprob_factor, omega, O_dist
#		int frame, i, j, O_ind, O2_ind, O_before, O_after, jumper, covsize
#	covsize = covevo.shape[1]
#	start_time = time.time()
#	jumpprob_factor = float(Hs.shape[1])*(Os.shape[1]-Hs.shape[1])/(Os.shape[1]*(Os.shape[1]-1))
#	for frame in xrange(Os.shape[0]-1):
#		if verbose == True:
#			if frame % 100 == 0:
#				print "#Frame {:8}, {:8.2f} fps".format(frame, float(frame)/(time.time()-start_time)), "\r",
#		jumper = neighborchange(covevo[frame], covevo[frame+1])
#		if jumper != -1:
#			O_before = covevo[frame, jumper]
#			O_after = covevo[frame+1, jumper]
#			O_dist = npa.length_ptr(&Os[frame, O_after, 0], &Os[frame, O_before, 0], &pbc[0])			
#			jump_prob = fermi(a, b, c, O_dist)
#			prob *= jump_prob
#		else:
			##calculate probability for no jump at all
			##calculate complementary event: any jump
#			omega = 0
#			for i in range(covevo.shape[1]):
#				for j in range(Os.shape[1]):
#					if contains_ptr(j, &covevo[frame, 0], covsize) == 0:
#						omega += fermi(a, b, c, npa.length_ptr(&Os[frame, covevo[frame, i], 0], &Os[frame, j, 0], &pbc[0]))
#			prob *= 1 - omega
#	print ""
#	return prob

#~ @cython.cdivision(True)
#~ @cython.wraparound(False)
#~ @cython.boundscheck(False)	
#~ def evaluate_model(double [:] a_range, double [:] b_range, double [:] c_range, double [:, :, ::1] Os, double [:, :, ::1] Hs, cnp.int64_t [:, ::1] covevo, double [:] pbc, int frame, double [:, :, ::1] prob_anterior):
#~ 
#~ 	cdef: 
#~ 		int i, j, k, l, O2_ind, jump_proton, O_before, O_after, covsize
#~ 		double O_dist, omega
#~ 	covsize = covevo.shape[1]
#~ 	jump_proton = neighborchange_ptr(&covevo[frame, 0], &covevo[frame+1, 0], covsize) 
#~ 	if jump_proton != -1:
#~ 		O_before = covevo[frame, jump_proton]
#~ 		O_after = covevo[frame+1, jump_proton]
#~ 		O_dist = npa.length_ptr(&Os[frame, O_after, 0], &Os[frame, O_before, 0], &pbc[0])
#~ 		for i in range(a_range.shape[0]):
#~ 			for j in range(b_range.shape[0]):
#~ 				for k in range(c_range.shape[0]):
#~ 					prob_anterior[i,j,k] *= fermi(a_range[i], b_range[j], c_range[k], O_dist)
#~ 					#if prob_anterior[i,j,k] < 0:
#~ 						#ipdb.set_trace()
#~ 	else:
#~ 		#calculate probability for no jump at all
#~ 		#calculate complementary event: any jump
#~ 		for i in range(a_range.shape[0]):
#~ 			for j in range(b_range.shape[0]):
#~ 				for k in range(c_range.shape[0]):
#~ 					if prob_anterior[i,j,k] != 0:
#~ 						omega = 0
#~ 						for l in range(covevo.shape[1]):
#~ 							for O2_ind in range(Os.shape[1]):
#~ 								if contains_ptr(O2_ind, &covevo[frame, 0], covsize) == 0:
#~ 									omega += fermi(a_range[i], b_range[j], c_range[k], npa.length_ptr(&Os[frame, covevo[frame, l], 0], &Os[frame, O2_ind, 0], &pbc[0]))
#~ 									#if verbose == True and prob > 1e-10:
#~ 										#print "#Jump possible from {} to {} with probability {}".format(O_ind, O2_ind, prob)
#~ 						#if verbose == True:
#~ 							#print "#({} {} {}) Total probability for a jump: {}".format(a, b, c, omega)
#~ 						prob_anterior[i,j,k] *= (1 - omega)
#~ 						if prob_anterior[i,j,k] < 0:
#~ 							prob_anterior[i,j,k] = 0
#~ @cython.cdivision(True)
#~ @cython.wraparound(False)
#~ @cython.boundscheck(False)	
#~ def eval_complete(double [:] a_range, double [:] b_range, double [:] c_range, double [:, :, ::1] Os, double [:, :, ::1] Hs, cnp.int64_t [:, ::1] covevo, double [:] pbc, double [:, :, ::1] prob_anterior, int verbose=0):
#~ 
#~ 	cdef: 
#~ 		int i, j, k, l, O2_ind, jump_proton, O_before, O_after, covsize, frame
#~ 		double O_dist, omega
#~ 	covsize = covevo.shape[1]
#~ 	start_time = time.time()
#~ 	for frame in range(Os.shape[0]-1):
#~ 		jump_proton = neighborchange_ptr(&covevo[frame, 0], &covevo[frame+1, 0], covsize) 
#~ 		if jump_proton != -1:
#~ 			O_before = covevo[frame, jump_proton]
#~ 			O_after = covevo[frame+1, jump_proton]
#~ 			O_dist = npa.length_ptr(&Os[frame, O_after, 0], &Os[frame, O_before, 0], &pbc[0])
#~ 			for i in range(a_range.shape[0]):
#~ 				for j in range(b_range.shape[0]):
#~ 					for k in range(c_range.shape[0]):
#~ 						prob_anterior[i,j,k] *= fermi(a_range[i], b_range[j], c_range[k], O_dist)
#~ 						#if prob_anterior[i,j,k] < 0:
#~ 							#ipdb.set_trace()
#~ 		else:
#~ 			#calculate probability for no jump at all
#~ 			#calculate complementary event: any jump
#~ 			for i in range(a_range.shape[0]):
#~ 				for j in range(b_range.shape[0]):
#~ 					for k in range(c_range.shape[0]):
#~ 						if prob_anterior[i,j,k] != 0:
#~ 							omega = 0
#~ 							for l in range(covevo.shape[1]):
#~ 								for O2_ind in range(Os.shape[1]):
#~ 									if contains_ptr(O2_ind, &covevo[frame, 0], covsize) == 0:
#~ 										omega += fermi(a_range[i], b_range[j], c_range[k], npa.length_ptr(&Os[frame, covevo[frame, l], 0], &Os[frame, O2_ind, 0], &pbc[0]))
#~ 										#if verbose == True and prob > 1e-10:
#~ 											#print "#Jump possible from {} to {} with probability {}".format(O_ind, O2_ind, prob)
#~ 							#if verbose == True:
#~ 								#print "#({} {} {}) Total probability for a jump: {}".format(a, b, c, omega)
#~ 							prob_anterior[i,j,k] *= (1 - omega)
#~ 							if prob_anterior[i,j,k] < 0:
#~ 								prob_anterior[i,j,k] = 0
#~ 		print float(frame)/(time.time()-start_time), "\r",
