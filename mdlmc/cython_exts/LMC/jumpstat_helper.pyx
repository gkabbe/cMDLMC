# cython: profile = False
# cython: boundscheck = False, wraparound = False, cdivision = True, initializedcheck = False
# cython: language_level = 3
# TODO: Cleanup
import numpy as np
import sys
import time
import cython

cimport numpy as cnp
cimport cython

from libcpp.vector cimport vector

from mdlmc.cython_exts.atoms cimport numpyatom as cnpa

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil


cdef inline int contains_ptr(cnp.int64_t x, cnp.int64_t * vec, int size):
    cdef int i
    for i in range(size):
        if x == vec[i]:
            return 1
    return 0


cdef int contains(cnp.int64_t x, cnp.int64_t [:] vec):
    cdef int i
    for i in range(vec.shape[0]):
        if x == vec[i]:
            return 1
    return 0


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
                        distance = cnpa.length_nonortho_bruteforce_ptr(&Os[i, O_i, 0],
                                                                       &Os[i, O_j, 0], &h[0, 0],
                                                                       &h_inv[0, 0])
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
                print("# {:} {:.2f} fps".format(i, i/(time.time()-start_time)), end="\r")
    print("")

    print("# Oxygen distance, jump probability, jump probability standard deviation")
    for i in xrange(bins):
        if len(probhisto[i]) > 0:
            print(dmin + (dmax - dmin) / bins / 2 + i * (dmax - dmin) / bins,
                  np.array(probhisto[i]).mean(), np.sqrt(np.array(probhisto[i]).var()))
