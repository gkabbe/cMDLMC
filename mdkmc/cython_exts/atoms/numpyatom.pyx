#cython: boundscheck=False, wraparound=False, boundscheck=False, cdivision=True
#cython: initializedcheck=False, profile=False
# TODO: Cleanup
import numpy as np
cimport numpy as np

from cython_gsl cimport round
from mdkmc.cython_exts.helper cimport math_helper as mh
from libcpp.vector cimport vector

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil


xyzatom = np.dtype([("name", np.str_, 1), ("pos", np.float64, (3,))])


cdef void diff(double [:] a1_pos, double [:] a2_pos, double [:] pbc, double [:] diffvec):
    cdef:
        int i
    for i in range(3):
        diffvec[i] = a2_pos[i] - a1_pos[i]
        while diffvec[i] < -pbc[i]/2:
            diffvec[i] += pbc[i]
        while diffvec[i] > pbc[i]/2:
            diffvec[i] -= pbc[i]


cdef void diff_ptr(double *a1_pos, double *a2_pos, double *pbc, double *diffvec) nogil:
    cdef:
        int i

    for i in range(3):
        diffvec[i] = a2_pos[i] - a1_pos[i]
        while diffvec[i] < -pbc[i]/2:
            diffvec[i] += pbc[i]
        while diffvec[i] > pbc[i]/2:
            diffvec[i] -= pbc[i]


cdef void diff_nonortho(double [:] a1_pos, double [:] a2_pos, double [:] diffvec,
                        double [:,::1] h, double [:,::1] h_inv):
    cdef:
        int i

    for i in range(3):
        diffvec[i] = a2_pos[i] - a1_pos[i]

    mh.matrix_mult(h_inv, diffvec)

    for i in range(3):
        diffvec[i] -= round(diffvec[i])

    mh.matrix_mult(h, diffvec)


cdef void diff_ptr_nonortho(double  *a1_pos, double *a2_pos, double *diffvec, double * h,
                            double * h_inv) nogil:
    cdef:
        int i

    for i in range(3):
        diffvec[i] = a2_pos[i] - a1_pos[i]

    mh.matrix_mult_ptr(h_inv, diffvec)

    for i in range(3):
        diffvec[i] -= round(diffvec[i])

    mh.matrix_mult_ptr(h, diffvec)


cpdef double length_nonortho_bruteforce(double [:] a1_pos, double [:] a2_pos, double [:,::1] h,
                                        double [:,::1] h_inv):
    cdef:
        double diffvec[3]
        vector[double] d
        vector[vector[double]] dists
        int i,j,k,dim
        double mindist = 1e6, dist

    diff_ptr_nonortho(&a1_pos[0], &a2_pos[0], &diffvec[0], &h[0, 0], &h_inv[0, 0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                d.clear()
                for dim in range(3):
                    d.push_back(diffvec[dim] + i * h[dim, 0] + j * h[dim, 1] + k * h[dim, 2])
                dists.push_back(d)
    for i in range(dists.size()):
        dist = dists[i][0] * dists[i][0] + dists[i][1] * dists[i][1] + dists[i][2] * dists[i][2]
        if dist < mindist:
            mindist = dist
    return sqrt(mindist)


cdef double length_nonortho_bruteforce_ptr(double *a1_pos, double *a2_pos, double *h,
                                           double *h_inv) nogil:
    cdef:
        double diffvec[3]
        vector[double] d
        vector[vector[double]] dists
        int i, j, k, dim
        double mindist = 1e6, dist

    diff_ptr_nonortho(a1_pos, a2_pos, diffvec, h, h_inv)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                d.clear()
                for dim in range(3):
                    d.push_back(
                        diffvec[dim] + i * h[3 * dim] + j * h[3 * dim + 1] + k * h[3 * dim + 2])
                dists.push_back(d)
    for i in range(dists.size()):
        dist = dists[i][0] * dists[i][0] + dists[i][1] * dists[i][1] + dists[i][2] * dists[i][2]
        if dist < mindist:
            mindist = dist
    return sqrt(mindist)


cdef diff_nonortho_bruteforce(double [:] a1_pos, double [:] a2_pos, double [:] diffvec,
                              double [:,::1] h, double [:,::1] h_inv):
    cdef:
        vector[double] d
        vector[vector[double]] dists
        int i, j, k, dim, index
        double mindist = 1e6, dist

    diff_ptr_nonortho(&a1_pos[0], &a2_pos[0], &diffvec[0], &h[0, 0], &h_inv[0, 0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                d.clear()
                for dim in range(3):
                    d.push_back(diffvec[dim] + i * h[dim, 0] + j * h[dim, 1] + k * h[dim, 2])
                dists.push_back(d)
    for i in range(dists.size()):
        dist = dists[i][0] * dists[i][0] + dists[i][1] * dists[i][1] + dists[i][2] * dists[i][2]
        if dist < mindist:
            mindist = dist
            index = i
    diffvec[0] = dists[index][0]
    diffvec[1] = dists[index][1]
    diffvec[2] = dists[index][2]


def bruteforce_test(double [:] a1_pos, double [:] a2_pos, double [:,::1] h, double [:,::1] h_inv):
    diffvec = np.zeros(3)
    l1 = length_nonortho_bruteforce(a1_pos, a2_pos, h, h_inv)
    diff_nonortho_bruteforce(a1_pos, a2_pos, diffvec, h, h_inv)

    print l1, np.linalg.norm(diffvec)


cpdef double length(double [:] a1_pos, double [:] a2_pos, double [:] pbc):
    cdef:
        np.ndarray[np.float_t, ndim=1] dist = np.zeros(3, float)
        int i

    diff_ptr(&a1_pos[0], &a2_pos[0], &pbc[0], &dist[0])

    return sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])


cdef double length_ptr(double *a1_pos, double *a2_pos, double *pbc) nogil:
    cdef:
        double dist[3]
        int i

    diff_ptr(a1_pos, a2_pos, pbc, &dist[0])

    return sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])
        

cpdef double sqdist(double [:] a1_pos, double [:] a2_pos, double [:] pbc):
    cdef:
        double dist[3]
        int i
    diff_ptr(&a1_pos[0], &a2_pos[0], &pbc[0], &dist[0])

    return dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]


def next_neighbor(double [:] a1_pos, double [:, ::1] atoms_pos, double [:] pbc, *,
                  exclude_identical_position=False):
    "search for nearest neighbor and return its index and distance"
    cdef:
        double mindist = 1e6
        int minind = -1
        int i
        double dist = -1

    for i in range(atoms_pos.shape[0]):
        dist = sqdist(a1_pos, atoms_pos[i], pbc)
        if dist < mindist and (dist > 0 or exclude_identical_position == False):
            mindist = dist
            minind = i
    return minind, mindist


cdef int next_neighbor_index(double [:] a1_pos, double [:, ::1] atoms_pos, double [:] pbc, *,
                  exclude_identical_position=False):
    "search for nearest neighbor and return its index and distance"
    cdef:
        double mindist = 1e6
        int minind = -1
        int i
        double dist = -1

    for i in range(atoms_pos.shape[0]):
        dist = sqdist(a1_pos, atoms_pos[i], pbc)
        if dist < mindist and (dist > 0 or exclude_identical_position == False):
            mindist = dist
            minind = i
    return minind, mindist


def next_neighbor_nonortho(double [:] a1_pos, double [:, ::1] atoms_pos, double [:,::1] h, 
                           double [:,::1] h_inv):
    "search for nearest neighbor and return its index and distance. For nonorthogonal boxes"
    cdef:
        double mindist = 1e6
        int minind = -1
        int i
        double diff[3]

    for i in range(atoms_pos.shape[0]):
        dist = length_nonortho_bruteforce(a1_pos, atoms_pos[i], h, h_inv)
        if dist < mindist:
            mindist = dist
            minind = i
    return minind, mindist


cpdef double angle(double [:] a1, double [:] a2, double [:] a3, double [:] a4, double [:] pbc):
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
    return acos(mh.dot_product_ptr(&v1[0], &v2[0], 3) / sqrt(mh.dot_product_ptr(&v1[0], &v1[0], 3))\
        / sqrt(mh.dot_product_ptr(&v2[0], &v2[0], 3)))


cdef double angle_ptr(double * a1, double * a2, double * a3, double * a4, double * pbc) nogil:
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
    return acos(mh.dot_product_ptr(&v1[0], &v2[0], 3) / sqrt(mh.dot_product_ptr(&v1[0], &v1[0], 3))\
        / sqrt(mh.dot_product_ptr(&v2[0], &v2[0], 3)))


cpdef double angle_nonortho(double [:] a1, double [:] a2, double [:] a3, double [:] a4,
                            double [:,::1] h, double [:,::1] h_inv):
    cdef:
        double v1[3]
        double v2[3]
        int i

    diff_nonortho_bruteforce(a1, a2, v1, h, h_inv)
    diff_nonortho_bruteforce(a3, a4, v2, h, h_inv)

    return acos(mh.dot_product_ptr(v1, v2, 3) / sqrt(mh.dot_product_ptr(v1, v1, 3)) / \
                sqrt(mh.dot_product_ptr(v2, v2, 3)))

cdef double angle_ptr_nonortho(double * a1, double * a2, double * a3, double * a4, double * h,
                               double * h_inv) nogil:
    cdef:
        double v1[3]
        double v2[3]
        int i

    diff_ptr_nonortho(&a1[0], &a2[0], &v1[0], &h[0], &h_inv[0])
    diff_ptr_nonortho(&a3[0], &a4[0], &v2[0], &h[0], &h_inv[0])

    return acos(mh.dot_product_ptr(v1, v2, 3) / sqrt(mh.dot_product_ptr(v1, v1, 3)) / \
                sqrt(mh.dot_product_ptr(v2, v2, 3)))
