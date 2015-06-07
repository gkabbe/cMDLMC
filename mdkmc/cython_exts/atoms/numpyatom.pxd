import numpy as np
cimport numpy as np
cimport cython

cpdef double length(double [:] a1_pos, double [:] a2_pos, double [:] pbc)

cdef double length_ptr(double *a1_pos, double *a2_pos, double *pbc) nogil

#~ cpdef double length_nonortho(double [:] a1_pos, double [:] a2_pos, double [:,::1] h, double [:,::1] h_inv)

#~ cdef double length_ptr_nonortho(double * a1_pos, double * a2_pos, double * h, double * h_inv)

cpdef double length_nonortho_bruteforce(double [:] a1_pos, double [:] a2_pos, double [:,::1] h, double [:,::1] h_inv)

cdef double length_nonortho_bruteforce_ptr(double * a1_pos, double * a2_pos, double * h, double * h_inv)

cdef void diff_ptr(double *a1_pos, double *a2_pos, double *pbc, double *diffvec) nogil

cdef void diff_ptr_nonortho(double  *a1_pos, double *a2_pos, double *diffvec, double * h, double * h_inv) nogil

cdef double angle(double [:] a1, double [:] a2, double [:] a3, double [:] a4, double [:] pbc)

cdef double angle_ptr(double * a1, double * a2, double * a3, double * a4, double * pbc) nogil

cpdef double angle_nonortho(double [:] a1, double [:] a2, double [:] a3, double [:] a4, double [:,::1] h, double [:,::1] h_inv)

cdef double angle_ptr_nonortho(double * a1, double * a2, double * a3, double * a4, double * h, double * h_inv) nogil


cdef test()
