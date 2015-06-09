#cython: profile=False
# TODO: Cleanup
import time
import numpy as np
import types
import logging

cimport numpy as np
cimport cython
from cython_gsl cimport *
from libc.stdlib cimport malloc, free

from libcpp.vector cimport vector
from libcpp.map cimport map

cimport mdkmc.cython_exts.atoms.numpyatom as cnpa

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil

from libc.stdio cimport *
# from libc.stdlib cimport bsearch
# from cython.parallel import prange

ctypedef np.int_t DTYPE_t

cdef double PI = np.pi

@cython.cdivision(True)
cdef double reservoir_function(double x, double width) nogil:
    return 0.5*(1+cos(x/width*PI))

@cython.cdivision(True)
cdef double dotprod_ptr(double* v1, double* v2) nogil:
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

#~ @cython.cdivision(True)
#~ cdef double dotprod(double_array_1d v1, double_array_1d v2):
#~ 	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdist(double [:] v1, double [:] v2, double [:] pbc) nogil:
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(3):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdist_no_z(double [:] v1, double [:] v2, double [:] pbc) nogil:
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(2):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef posdiff_no_z(vector[double] &posdiff, double [:] v1, double [:] v2, double [:] pbc):
    cdef int i
    for i in range(3):
        posdiff.push_back(v2[i]-v1[i])
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]

#~ cdef double angle(float *O1, float *P1, float *O2, float *P2, double* pbc):
#~ 	cdef:
#~ 		double v1[3]
#~ 		double v2[3]
#~ 		int i
#~ 	
#~ 	for i in range(3):
#~ 		v1[i] = O1[i] - P1[i]
#~ 		v2[i] = O2[i] - P2[i]
#~ 		
#~ 		while v1[i] > pbc[i]/2:
#~ 			v1[i] -= pbc[i]
#~ 		while v1[i] < -pbc[i]/2:
#~ 			v1[i] += pbc[i]
#~ 			
#~ 		while v2[i] > pbc[i]/2:
#~ 			v2[i] -= pbc[i]
#~ 		while v2[i] < -pbc[i]/2:
#~ 			v2[i] += pbc[i]
#~ 	return acos(dotprod(v1, v2)/dotprod(v1, v1)/dotprod(v2, v2))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double angle_ptr(double *a1, double *a2, double *a3, double *a4, double* pbc) nogil:
    cdef:
        double v1[3]
        double v2[3]
        int i

    for i in range(3):
        v1[i] = a2[i] - a1[i]
        v2[i] = a4[i] - a3[i]

        while v1[i] > pbc[i]/2:
            v1[i] -= pbc[i]
        while v1[i] < -pbc[i]/2:
            v1[i] += pbc[i]

        while v2[i] > pbc[i]/2:
            v2[i] -= pbc[i]
        while v2[i] < -pbc[i]/2:
            v2[i] += pbc[i]
    return acos(dotprod_ptr(v1, v2)/sqrt(dotprod_ptr(v1, v1))/sqrt(dotprod_ptr(v2, v2)))

#~ cdef double angle(float_or_double_array_1d O1, float_or_double_array_1d P1, float_or_double_array_1d O2, float_or_double_array_1d P2, double [:] pbc):

# def angle_test():
#     cdef:
#         double *pbc = [100, 100, 100]
#         double *O1 = [1, 0, 0]
#         double *P1 = [0, 0, 0]
#         double O2[3]
#
#     for i in xrange(100):
#         O2[0] = cos(2*np.pi/100*i)
#         O2[1] = sin(2*np.pi/100*i)
#         O2[2] = 0
#         print angle_ptr(&O1[0], &P1[0], &O2[0], &P1[0], &pbc[0])



cdef double angle_factor(double angle):
    if angle > PI/2:
        return cos(angle - PI)
    else:
        return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def dist(double [:] v1, double [:] v2, double [:] pbc):
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(3):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])

#~ @cython.cdivision(True)
#~ @cython.wraparound(False)
#~ @cython.boundscheck(False)
#~ def cget_ratesum_nointra(double [:, ::1] Opos, double [:] pbc, double a, double b, double c, double threshold=1e-8):
#~ 	cdef int i,j
#~ 	cdef double omega_framesum = 0
#~ 	cdef double om
#~ 	for i in range(Opos.shape[0]):
#~ 		for j in range(Opos.shape[0]):
#~ 			if i/3 != j/3:
#~ 				om = fermi(a, b, c, dist(Opos[i,:], Opos[j,:], pbc))
#~ 				if om >= threshold:
#~ 					omega_framesum += om
#~ 	return omega_framesum

#~ def calculate_transitionmatrix_nointra(double [:, ::1] transitionmatrix, double [:,::1] atom_positions, double[:] parameters, double timestep, double [:] pbc):
#~ 	cdef int i,j 
#~ 	cdef double a = parameters[0] * timestep
#~ 	for i in xrange(atom_positions.shape[0]):
#~ 		for j in xrange(atom_positions.shape[0]):
#~ 			if i/3 != j/3:
#~ 				transitionmatrix[i,j] = fermi(a, parameters[1], parameters[2], cdist(atom_positions[i], atom_positions[j], pbc))

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef read_Opos(double [:, ::1] Oarr, datei, int atoms):
    datei.readline()
    datei.readline()

    cdef int index = 0
    cdef int i = 0
    for i in range(atoms):
        line = datei.readline()
        if line.split()[0] == "O":
            Oarr[index, 0] = float(line.split()[1])
            Oarr[index, 1] = float(line.split()[2])
            Oarr[index, 2] = float(line.split()[3])
            index = index + 1

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def dist_numpy_all_inplace(double [:, ::1] displacement, double [:, ::1] arr1, double [:, ::1] arr2, double [:] pbc):
    cdef int i, j
    for i in xrange(arr1.shape[0]):
        for j in xrange(3):
            displacement[i, j] = arr2[i,j] - arr1[i,j]
            while displacement[i,j] > pbc[j]/2:
                displacement[i,j] -= pbc[j]
            while displacement[i,j] < -pbc[j]/2:
                displacement[i,j] += pbc[j]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def dist_numpy_all(double [:, ::1] arr1, double [:, ::1] arr2, double [:] pbc):
    cdef:
        int i, j
        double [:, ::1] displacement = np.zeros((arr1.shape[0], 3))
    for i in xrange(arr1.shape[0]):
        for j in xrange(3):
            displacement[i, j] = arr2[i,j] - arr1[i,j]
            while displacement[i,j] > pbc[j]/2:
                displacement[i,j] -= pbc[j]
            while displacement[i,j] < -pbc[j]/2:
                displacement[i,j] += pbc[j]
    return displacement

@cython.wraparound(False)
@cython.boundscheck(False)
def dist_numpy_all_nonortho(double [:, ::1] displacement, double [:, ::1] arr1, double [:, ::1] arr2, double [:,::1] h, double [:,::1] h_inv):
    cdef int i, j
    for i in xrange(arr1.shape[0]):
        cnpa.diff_ptr_nonortho(&arr1[i,0], &arr2[i,0], &displacement[i,0], &h[0,0], &h_inv[0,0])

def list_to_vector(l):
    cdef vector[vector[int]] v

    v=l

    for i in range(v.size()):
        for j in range(v[i].size()):
            print v[i][j],
        print ""

@cython.wraparound(False)
@cython.boundscheck(False)
def extend_simulationbox_z(int zfac, double [:,::1] Opos, double pbc_z, int oxygennumber):
    cdef int i,j,z
    for z in xrange(1, zfac):
        for i in xrange(oxygennumber):
            Opos[z*oxygennumber+i, 0] = Opos[i, 0]
            Opos[z*oxygennumber+i, 1] = Opos[i, 1]
            Opos[z*oxygennumber+i, 2] = Opos[i, 2] + pbc_z*z

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def extend_simulationbox(double [:, :, :, :, ::1] Opos, double [:,::1] h, int multx, int multy, int multz, int initial_oxynumber):
    cdef int x,y,z,i,j
    for x in range(multx):
        for y in range(multy):
            for z in range(multz):
                if x+y+z != 0:
                    for i in range(initial_oxynumber):
                        for j in range(3):
                            Opos[x, y, z, i, j] = Opos[0, 0, 0, i, j] + x * h[0,j] + y * h[1,j] + z * h[2,j]

def count_protons_and_oxygens(double[:,::1] Opos, np.uint8_t [:] proton_lattice, np.int64_t [:] O_counter, np.int64_t [:] H_counter, double [:] bin_bounds):
    cdef:
        int i, O_index
        double O_z

    for i in xrange(proton_lattice.shape[0]):
        O_z = Opos[i,2]
        O_index = np.searchsorted(bin_bounds, O_z)-1
        O_counter[O_index] += 1
        if proton_lattice[i] > 0:
            H_counter[O_index] += 1

#Define Function Objects. These can later be substituted easily by kMC_helper
cdef class Function:
    cdef double evaluate(self, double x):
        return 0

cdef class FermiFunction(Function):
    cdef:
        double a, b, c
    def __cinit__(self, double a, double b, double c):
        self.a = a
        self.b = b
        self.c = c

    @cython.cdivision(True)
    cdef double evaluate(self, double x):
        return self.a/(1+exp((x-self.b)/self.c))

cdef class AEFunction(Function):
    cdef:
        double A, a, x0, xint
    def __cinit__(self, double A, double a, double x0, double xint):
        self.a = a
        self.x0 = x0
        self.xint = xint

    @cython.cdivision(True)
    cdef double evaluate(self, double x):
        if x <= self.x0:
            return 0
        elif self.x0 < x < self.xint:
            return self.a*(x-self.x0)*(x-self.x0)
        else:
            return 2*self.a*(self.xint-self.x0)*(x-self.xint) + self.a*(self.xint-self.x0)*(self.xint-self.x0)

cdef class Helper:
    cdef:
        gsl_rng * r
        int jumps
        int matlen
        int nonortho
        vector[int] start
        vector[int] dest
        vector[double] prob
        vector[vector[double]] jumpdists
        vector[vector[int]] neighbors
        double [:] pbc
        double [:,::1] pbc_nonortho
        double [:,::1] h
        double [:,::1] h_inv
        object logger
        Function jumprate

    def __cinit__(self, np.ndarray [np.double_t, ndim=1] pbc, int nonortho,
                  jumprate_parameter_dict, jumprate_type="MD_rates",
                  seed=None, verbose=False):
        self.logger = logging.getLogger("{}.{}.{}".format("__main__.MDMC",
                                                          __name__,
                                                          self.__class__.__name__))
        self.logger.info("Creating an instance of {}.{}.{}".format("MDMC",
                                                                   __name__,
                                                                   self.__class__.__name__))
        self.r = gsl_rng_alloc(gsl_rng_mt19937)
        self.jumps = 0
        self.nonortho = nonortho
        if nonortho == 0:
            self.pbc = pbc
        else:
            self.pbc_nonortho = pbc.reshape((3,3))
            self.h = np.array(pbc.reshape((3,3)).T, order="C")
            self.h_inv = np.array(np.linalg.inv(self.h), order="C")

        if type(seed) != types.IntType:
            seed = time.time()
        gsl_rng_set(self.r, seed)
        if verbose == True:
            print "#Using seed", seed

        if jumprate_type == "MD_rates":
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            c = jumprate_parameter_dict["c"]
            self.jumprate = FermiFunction(a, b, c)

        elif jumprate_type == "AE_rates":
            A = jumprate_parameter_dict["A"]
            a = jumprate_parameter_dict["a"]
            x0 = jumprate_parameter_dict["x0"]
            xint = jumprate_parameter_dict["xint"]
            self.jumprate = AEFunction(A, a, x0, xint)
        else:
            raise Exception("Jumprate type unknown. Please choose between "
                            "MD_rates and AE_rates")

    def __dealloc__(self):
        gsl_rng_free(self.r)

    def determine_neighbors(self, double [:, ::1] Opos_large, double r_cut, verbose=False):
        cdef:
            int i,j
            double dist
            vector[int] a
        self.neighbors.clear()
        if self.nonortho == 0:
            for i in xrange(Opos_large.shape[0]):
                a.clear()
                for j in xrange(Opos_large.shape[0]):
                    if i != j:
                        dist = cnpa.length_ptr(&Opos_large[i,0], &Opos_large[j,0], &self.pbc[0])
                        if dist < r_cut:
                            a.push_back(j)
                self.neighbors.push_back(a)
        else:
            for i in xrange(Opos_large.shape[0]):
                a.clear()
                for j in xrange(Opos_large.shape[0]):
                    if i != j:
                        dist = cnpa.length_nonortho_bruteforce_ptr(&Opos_large[i,0], &Opos_large[j,0],
                                                                   &self.h[0,0], &self.h_inv[0,0])
                        if dist < r_cut:
                            a.push_back(j)
                self.neighbors.push_back(a)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calculate_transitions_new(self, double [:,::1] Os, double r_cut):
        cdef:
            int i,j
            double dist

        self.start.clear()
        self.dest.clear()
        self.prob.clear()
        for i in range(Os.shape[0]):
            for j in range(self.neighbors[i].size()):
                # if i/3 != self.neighbors[i][j]/3: #only for hexaphenylbenzene (forbids jumps within phosphonic group)
#~ 				dist = cnpa.length_ptr(&Os[i,0], &Os[self.neighbors[i][j],0], &self.pbc[0])
                dist = cnpa.length(Os[i], Os[self.neighbors[i][j]], self.pbc)
                if dist < r_cut:
#~ 					prob = fermi(parameters[0], parameters[1], parameters[2], dist)
#~ 					self.logger.debug("Adding Transition {}-{} with distance {:.4f} and probability {:.4f}".format(i,self.neighbors[i][j],dist,prob))
                    self.start.push_back(i)
                    self.dest.push_back(self.neighbors[i][j])
                    # self.prob.push_back(fermi(parameters[0], parameters[1], parameters[2], dist))
                    self.prob.push_back(self.jumprate.evaluate(dist))

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calculate_transitions_POOangle(self, double [:,::1] Os, double [:,::1] Ps, np.int64_t [:] P_neighbors, double r_cut, double angle_thresh):
        cdef:
            int i,j, index2
            double dist, PO_angle
        self.start.clear()
        self.dest.clear()
        self.prob.clear()
        for i in range(Os.shape[0]):
            for j in range(self.neighbors[i].size()):
                index2 = self.neighbors[i][j]
                if self.nonortho == 0:
                    dist = cnpa.length_ptr(&Os[i,0], &Os[index2,0], &self.pbc[0])
#~ 					dist = cnpa.length(Os[i], Os[self.neighbors[i][j]], self.pbc)
                else:
#~ 					dist = cnpa.length_nonortho_bruteforce_ptr(&Os[i,0], &Os[index2,0], &self.h[0,0], &self.h_inv[0,0])
                    dist = cnpa.length_nonortho_bruteforce(Os[i], Os[index2], self.h, self.h_inv)
                if dist < r_cut:
#~ 					if dist == 0:
#~ 						print "dist=0 for i=", i, ",j=", index2
                    if self.nonortho == 0:
                        # POO_angle = cnpa.angle(Os[i], Ps[P_neighbors[i]], Os[i], Os[index2], self.pbc)
                        POO_angle = cnpa.angle_ptr(&Os[i, 0], &Ps[P_neighbors[i], 0], &Os[i, 0], &Os[index2, 0], &(self.pbc)[0])
                    else:
#~ 						POO_angle = cnpa.angle_nonortho(Os[i], Ps[P_neighbors[i]], Os[i], Os[index2], self.h, self.h_inv)
                        POO_angle = cnpa.angle_nonortho(Os[i], Ps[P_neighbors[i]], Os[i], Os[index2], self.h, self.h_inv)
                    self.start.push_back(i)
                    self.dest.push_back(self.neighbors[i][j])
                    if POO_angle >= angle_thresh:
                        # self.prob.push_back(fermi(parameters[0], parameters[1], parameters[2], dist))
                        self.prob.push_back(self.jumprate.evaluate(dist))
                    else:
                        self.prob.push_back(0)

#use only for hexaphenyl!
#~ 	@cython.cdivision(True)
#~ 	@cython.wraparound(False)
#~ 	@cython.boundscheck(False)
#~ 	def calculate_transitions_reservoir(self, double [:,::1] Os, double [:] parameters, double [:] pbc, double r_cut, double res_width):
#~ 		cdef:
#~ 			int i,j,k
#~ 			double dist, zmin, zmax
#~ 			vector[double] posdiff
#~ 
#~ 		self.start.clear()
#~ 		self.dest.clear()
#~ 		self.prob.clear()
#~ 		self.jumpdists.clear()
#~ 		zmin = 1e10
#~ 		zmax = -1e10
#~ 
#~ 		for i in range(Os.shape[0]):
#~ 			# print i
#~ 			if Os[i,2] > zmax:
#~ 				zmax = Os[i,2]
#~ 			if Os[i,2] < zmin:
#~ 				zmin = Os[i,2]
#~ 		# print "zmin:", zmin, "zmax:", zmax
#~ 
#~ 		for i in range(Os.shape[0]):
#~ 			for j in range(self.neighbors[i].size()):
#~ 				if i/3 != self.neighbors[i][j]/3:
#~ 					dist = cdist_no_z(Os[i,:], Os[self.neighbors[i][j],:], pbc)
#~ 					# print dist, i, self.neighbors[i][j]
#~ 					if dist < r_cut:
#~ 						# print "bond", i, self.neighbors[i][j]
#~ 						posdiff.clear()
#~ 						self.start.push_back(i)
#~ 						self.dest.push_back(self.neighbors[i][j])
#~ 						self.prob.push_back(fermi(parameters[0], parameters[1], parameters[2], dist))
#~ 						# print "posdiff_no_z"
#~ 						posdiff_no_z(posdiff, Os[i,:], Os[self.neighbors[i][j],:], pbc)
#~ 						self.jumpdists.push_back(posdiff)
#~ 
#~ 			if Os[i,2] - zmin <= res_width:
#~ 				# print "source"
#~ 				posdiff.clear()
#~ 				self.start.push_back(-1)
#~ 				self.dest.push_back(i)
#~ 				self.prob.push_back(reservoir_function(Os[i,2] - zmin, res_width))
#~ 				
#~ 				posdiff.resize(3,0)
#~ 				
#~ 				self.jumpdists.push_back(posdiff)
#~ 
#~ 			if zmax - Os[i,2] <= res_width:
#~ 				# print "drain"
#~ 				posdiff.clear()
#~ 				self.start.push_back(i)
#~ 				self.dest.push_back(-1)
#~ 				self.prob.push_back(reservoir_function(zmax - Os[i,2], res_width))
#~ 		
#~ 				posdiff.resize(3,0)
#~ 				
#~ 				self.jumpdists.push_back(posdiff)


#~ 	def print_transitions(self):
#~ 		cdef int i
#~ 		for i in range(self.start.size()):
#~ 			print self.start[i], self.dest[i], self.prob[i], self.jumpdists[i]

#~ 	@cython.cdivision(True)
#~ 	@cython.wraparound(False)
#~ 	@cython.boundscheck(False)
#~ 	def calculate_transitionsmatrix_reservoir(self, double [:,::1] tm, double [:,::1] Os, double [:] parameters, double [:] pbc, double r_cut, double res_width):
#~ 		cdef:
#~ 			int i,j,k
#~ 			double dist, zmin, zmax
#~ 
#~ 		zmin = 1e10
#~ 		zmax = -1e10
#~ 
#~ 		for i in range(Os.shape[0]):
#~ 			# print i
#~ 			if Os[i,2] > zmax:
#~ 				zmax = Os[i,2]
#~ 			if Os[i,2] < zmin:
#~ 				zmin = Os[i,2]
#~ 		# print "zmin:", zmin, "zmax:", zmax
#~ 
#~ 		for i in range(Os.shape[0]):
#~ 			for j in range(Os.shape[0]):
#~ 				if i/3 != j/3:
#~ 					dist = cdist_no_z(Os[i,:], Os[j,:], pbc)
#~ 					# print dist, i, self.neighbors[i][j]
#~ 					if dist < r_cut:
#~ 						tm[i+1,j+1] += fermi(parameters[0], parameters[1], parameters[2], dist)
#~ 
#~ 			if Os[i,2] - zmin <= res_width:
#~ 				tm[0,i+1] += reservoir_function(Os[i,2] - zmin, res_width)
#~ 			
#~ 			if zmax - Os[i,2] <= res_width:
#~ 				tm[i+1,tm.shape[1]-1] += reservoir_function(zmax - Os[i,2], res_width)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calculate_transitions_avg(self, double [:,::1] tm, double [:,::1] Opos_avg):
        cdef:
            int i,j
            vector[double] posdiff
        self.start.clear()
        self.dest.clear()
        self.prob.clear()
        self.jumpdists.clear()

        for i in range(1, tm.shape[0]-1):
            for j in range(1,tm.shape[1]-1):
                if tm[i,j] > 1e-8:
                    # print i,j
                    posdiff.clear()
                    self.start.push_back(i-1)
                    self.dest.push_back(j-1)
                    self.prob.push_back(tm[i,j])
                    posdiff_no_z(posdiff, Opos_avg[i-1,:], Opos_avg[j-1,:], self.pbc)
                    self.jumpdists.push_back(posdiff)
        #reservoir
        for i in range(1,tm.shape[0]-1):
            if tm[0,i] > 1e-8:
                # print "source",i
                for j in range(3):
                    posdiff.push_back(0)
                self.jumpdists.push_back(posdiff)
                self.start.push_back(-1)
                self.dest.push_back(i-1)
                self.prob.push_back(tm[0,i])
            if tm[i,tm.shape[0]-1] > 1e-8:
                # print "drain", i
                for j in range(3):
                    posdiff.push_back(0)
                self.jumpdists.push_back(posdiff)
                self.start.push_back(i-1)
                self.dest.push_back(-1)
                self.prob.push_back(tm[i,tm.shape[0]-1])



#~ 	@cython.cdivision(True)
#~ 	@cython.wraparound(False)
#~ 	@cython.boundscheck(False)
#~ 	def calculate_transitionsmatrix_old(self, double [:,::1] Os, double [:, ::1] tm, double [:] parameters, double [:] pbc, double r_cut):
#~ 		cdef:
#~ 			int i,j
#~ 			double dist
#~ 		for i in range(Os.shape[0]):
#~ 			for j in range(Os.shape[0]):
#~ 				if i != j:
#~ 					dist = cdist(Os[i], Os[j], pbc)
#~ 					if dist < r_cut:
#~ 						tm[i,j] += fermi(parameters[0], parameters[1], parameters[2], dist)
#~ 
#~ 	@cython.cdivision(True)
#~ 	@cython.wraparound(False)
#~ 	@cython.boundscheck(False)
#~ 	def calculate_transitionsmatrix_new(self, double [:,::1] Os, double [:, ::1] tm, double [:] parameters, double [:] pbc, double r_cut):
#~ 		cdef:
#~ 			int i,j,k
#~ 			double dist
#~ 		for i in range(Os.shape[0]):
#~ 			for j in range(self.neighbors[i].size()):
#~ 				if i/3 != j/3:
#~ 					k = self.neighbors[i][j]
#~ 					dist = cdist(Os[i,:], Os[k,:], pbc)
#~ 					if dist < r_cut:
#~ 						tm[i, k] += fermi(parameters[0], parameters[1], parameters[2], dist)

     # @cython.cdivision(True)
     # @cython.wraparound(False)
     # @cython.boundscheck(False)
     # def fermi_test(self, double [:] values, double [:] parameters):
     #     cdef int i
     #     for i in range(values.size):
     #         print fermi(parameters[0], parameters[1], parameters[2], values[i])

    def jumprate_test(self, double x):
        return self.jumprate.evaluate(x)

    def print_transitions(self):
        cdef int i
        for i in range(self.start.size()):
            print self.start[i], self.dest[i], self.prob[i]
        print "In total", self.start.size(), "connections"

    def return_transitions(self):
        return self.start, self.dest, self.prob

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sweep_gsl(self, np.uint8_t [:] proton_lattice, double [:,::1] transitionmatrix):
        cdef:
            int i,j, index_origin, index_destin
            int matlen = transitionmatrix.shape[0]
        for i in xrange(matlen):
            for j in xrange(matlen):
                index_origin = gsl_rng_uniform_int(self.r, matlen)
                index_destin = gsl_rng_uniform_int(self.r, matlen)
                if proton_lattice[index_origin] > 0 and proton_lattice[index_destin] == 0:
                    if gsl_rng_uniform(self.r) < transitionmatrix[index_origin,index_destin]:
                        proton_lattice[index_destin] = proton_lattice[index_origin]
                        proton_lattice[index_origin] = 0
                        self.jumps += 1
        return self.jumps

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sweep_list(self, np.uint8_t [:] proton_lattice):
        cdef:
            int i,j, index_origin, index_destin
            int steps = self.start.size()
        for j in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start[i]
            index_destin = self.dest[i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destin] == 0:
                if gsl_rng_uniform(self.r) < self.prob[i]:
                    proton_lattice[index_destin] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    self.jumps += 1
        return self.jumps

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sweep_list_jumpmat(self, np.uint8_t [:] proton_lattice, np.int64_t [:,::1] jumpmat):
        cdef:
            int i,j, index_origin, index_destin
            int steps = self.start.size()
        for j in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start[i]
            index_destin = self.dest[i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destin] == 0:
                if gsl_rng_uniform(self.r) < self.prob[i]:
                    proton_lattice[index_destin] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    jumpmat[index_destin, index_origin] += 1
                    self.jumps += 1
        return self.jumps

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sweep_list_reservoir(self, np.uint8_t [:] proton_lattice, double [:] r):
        cdef:
            int i,j, k, index_origin, index_destin
            int steps = self.start.size()

        for j in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start[i]
            index_destin = self.dest[i]

            if index_origin < 0 and gsl_rng_uniform(self.r) < self.prob[i]:
                # print "jump from source"
                proton_lattice[index_destin] = 1
            else:
                if index_destin < 0 and gsl_rng_uniform(self.r) < self.prob[i]:
                    # print "jump from", index_origin, "to drain"
                    proton_lattice[index_origin] = 0
                else:
                    if proton_lattice[index_origin] > 0 and proton_lattice[index_destin] == 0:
                        # print "jump from", index_origin, "to", index_destin
                        if gsl_rng_uniform(self.r) < self.prob[i]:
                            proton_lattice[index_destin] = proton_lattice[index_origin]
                            proton_lattice[index_origin] = 0
                            self.jumps += 1
                            # print "r +=", self.jumpdists[i]
                            for k in range(3):
                                r[k] += self.jumpdists[i][k]


        return self.jumps

    def reset_jumpcounter(self):
        self.jumps = 0

    def get_jumps(self):
        return self.jumps
