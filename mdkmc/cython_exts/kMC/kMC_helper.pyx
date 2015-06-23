#cython: profile=False
#cython: boundscheck=False, wraparound=False, boundscheck=False, cdivision=False, initializedcheck=False
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


cdef double R = 1.9872041e-3   # universal gas constant in kcal/mol/K


cdef double reservoir_function(double x, double width) nogil:
    return 0.5*(1+cos(x/width*PI))


cdef double dotprod_ptr(double* v1, double* v2) nogil:
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]


cdef double cdist(double [:] v1, double [:] v2, double [:] pbc) nogil:
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(3):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])


cdef double cdist_no_z(double [:] v1, double [:] v2, double [:] pbc) nogil:
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(2):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])


cdef posdiff_no_z(vector[double] &posdiff, double [:] v1, double [:] v2, double [:] pbc):
    cdef int i
    for i in range(3):
        posdiff.push_back(v2[i]-v1[i])
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]


# cdef double angle_ptr(double *a1, double *a2, double *a3, double *a4, double* pbc) nogil:
#     cdef:
#         double v1[3]
#         double v2[3]
#         int i
#
#     for i in range(3):
#         v1[i] = a2[i] - a1[i]
#         v2[i] = a4[i] - a3[i]
#
#         while v1[i] > pbc[i]/2:
#             v1[i] -= pbc[i]
#         while v1[i] < -pbc[i]/2:
#             v1[i] += pbc[i]
#
#         while v2[i] > pbc[i]/2:
#             v2[i] -= pbc[i]
#         while v2[i] < -pbc[i]/2:
#             v2[i] += pbc[i]
#     return acos(dotprod_ptr(v1, v2)/sqrt(dotprod_ptr(v1, v1))/sqrt(dotprod_ptr(v2, v2)))


cdef double angle_factor(double angle):
    if angle > PI/2:
        return cos(angle - PI)
    else:
        return 0

def dist(double [:] v1, double [:] v2, double [:] pbc):
    cdef int i
    cdef double *posdiff = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
    for i in range(3):
        while posdiff[i] > pbc[i]/2:
            posdiff[i] -= pbc[i]
        while posdiff[i] < -pbc[i]/2:
            posdiff[i] += pbc[i]
    return sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2])


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


def dist_numpy_all_inplace(double [:, ::1] displacement, double [:, ::1] arr1, double [:, ::1] arr2, double [:] pbc):
    cdef int i, j
    for i in xrange(arr1.shape[0]):
        for j in xrange(3):
            displacement[i, j] = arr2[i,j] - arr1[i,j]
            while displacement[i,j] > pbc[j]/2:
                displacement[i,j] -= pbc[j]
            while displacement[i,j] < -pbc[j]/2:
                displacement[i,j] += pbc[j]


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


def extend_simulationbox_z(int zfac, double [:,::1] Opos, double pbc_z, int oxygennumber):
    cdef int i,j,z
    for z in xrange(1, zfac):
        for i in xrange(oxygennumber):
            Opos[z*oxygennumber+i, 0] = Opos[i, 0]
            Opos[z*oxygennumber+i, 1] = Opos[i, 1]
            Opos[z*oxygennumber+i, 2] = Opos[i, 2] + pbc_z*z


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
    # cdef:
    #     double a, b, c
    def __cinit__(self, double a, double b, double c):
        self.a = a
        self.b = b
        self.c = c

    cdef double evaluate(self, double x):
        return self.a/(1+exp((x-self.b)/self.c))


cdef class AEFunction(Function):

    def __cinit__(self, double A, double a, double x0, double xint, T):
        self.A = A
        self.a = a
        self.x0 = x0
        self.xint = xint
        self.T = T

    cdef double evaluate(self, double x):
        cdef double E
        if x <= self.x0:
            E = 0
        elif self.x0 < x < self.xint:
            E = self.a * (x-self.x0) * (x-self.x0)
        else:
            E = 2*self.a * (self.xint-self.x0)*(x-self.xint) + self.a*(self.xint-self.x0)*(self.xint-self.x0)
        return self.A * exp(-E/(R*self.T))

cdef class AtomBox:

    cdef

cdef class Helper:
    cdef:
        gsl_rng * r
        int jumps
        int matlen
        int nonortho
        # Create containers for the oxygen index from which the proton jump start, for the index of
        # the destination oxygen, and for the jump probability of the oxygen connection
        # The _tmp vectors hold the information for a single frame, whereas the nested vectors hold
        # the indices and probabilities for the whole trajectory
        vector[vector[int]] start
        vector[vector[int]] destination
        vector[vector[double]] jump_probability
        vector[int] start_tmp
        vector[int] destination_tmp
        vector[double] jump_probability_tmp
        # vector[vector[double]] jumpdists
        vector[vector[int]] neighbors
        double [:] pbc
        double [:,::1] pbc_nonortho
        double [:,::1] h
        double [:,::1] h_inv
        object logger
        Function jumprate_fct

    def __cinit__(self, double [:] pbc, int nonortho,
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
            self.jumprate_fct = FermiFunction(a, b, c)

        elif jumprate_type == "AE_rates":
            A = jumprate_parameter_dict["A"]
            a = jumprate_parameter_dict["a"]
            x0 = jumprate_parameter_dict["x0"]
            xint = jumprate_parameter_dict["xint"]
            T = jumprate_parameter_dict["T"]
            self.jumprate_fct = AEFunction(A, a, x0, xint, T)
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

    def calculate_transitions_new(self, double [:,::1] Os, double r_cut):
        cdef:
            int i,j
            double dist

        self.start_tmp.clear()
        self.destination_tmp.clear()
        self.jump_probability_tmp.clear()
        for i in range(Os.shape[0]):
            for j in range(self.neighbors[i].size()):
                # if i/3 != self.neighbors[i][j]/3: #only for hexaphenylbenzene (forbids jumps within phosphonic group)
#~ 				dist = cnpa.length_ptr(&Os[i,0], &Os[self.neighbors[i][j],0], &self.pbc[0])
                dist = cnpa.length(Os[i], Os[self.neighbors[i][j]], self.pbc)
                if dist < r_cut:
#~ 					prob = fermi(parameters[0], parameters[1], parameters[2], dist)
#~ 					self.logger.debug("Adding Transition {}-{} with distance {:.4f} and probability {:.4f}".format(i,self.neighbors[i][j],dist,prob))
                    self.start_tmp.push_back(i)
                    self.destination_tmp.push_back(self.neighbors[i][j])
                    # self.prob.push_back(fermi(parameters[0], parameters[1], parameters[2], dist))
                    self.jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))

    def calculate_transitions_POOangle(self, double [:,::1] Os, double [:,::1] Ps, np.int64_t [:] P_neighbors, double r_cut, double angle_thresh):
        cdef:
            int i,j, index2
            double dist, PO_angle
        self.start_tmp.clear()
        self.destination_tmp.clear()
        self.jump_probability_tmp.clear()
        for i in range(Os.shape[0]):
            for j in range(self.neighbors[i].size()):
                index2 = self.neighbors[i][j]
                if self.nonortho == 0:
                    dist = cnpa.length_ptr(&Os[i,0], &Os[index2,0], &self.pbc[0])
                else:
                    dist = cnpa.length_nonortho_bruteforce(Os[i], Os[index2], self.h, self.h_inv)
                if dist < r_cut:
                    if self.nonortho == 0:
                        POO_angle = cnpa.angle_ptr(&Os[i, 0], &Ps[P_neighbors[i], 0], &Os[i, 0], &Os[index2, 0], &(self.pbc)[0])
                    else:
                        POO_angle = cnpa.angle_nonortho(Os[i], Ps[P_neighbors[i]], Os[i], Os[index2], self.h, self.h_inv)
                    self.start_tmp.push_back(i)
                    self.destination_tmp.push_back(self.neighbors[i][j])
                    if POO_angle >= angle_thresh:
                        self.jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))
                    else:
                        self.jump_probability_tmp.push_back(0)

        self.start.push_back(self.start_tmp)
        self.destination.push_back(self.destination_tmp)
        self.jump_probability.push_back(self.jump_probability_tmp)

    def get_transition_number(self):
        return self.start_tmp.size()

    def jumprate_test(self, double x):
        return self.jumprate_fct.evaluate(x)

    def print_transitions(self):
        cdef int i
        for i in range(self.start_tmp.size()):
            print self.start_tmp[i], self.destination_tmp[i], self.jump_probability_tmp[i]
        print "In total", self.start_tmp.size(), "connections"

    def return_transitions(self):
        return self.start_tmp, self.destination_tmp, self.jump_probability_tmp

    def sweep_list(self, np.uint8_t [:] proton_lattice):
        cdef:
            int i,j, index_origin, index_destin
            int steps = self.start_tmp.size()
        for j in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start_tmp[i]
            index_destin = self.destination_tmp[i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destin] == 0:
                if gsl_rng_uniform(self.r) < self.jump_probability_tmp[i]:
                    proton_lattice[index_destin] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    self.jumps += 1
        return self.jumps

    def reset_jumpcounter(self):
        self.jumps = 0

    def get_jumps(self):
        return self.jumps
