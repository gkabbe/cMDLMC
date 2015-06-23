import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating

ctypedef np.float32_t [:] float_array_1d_t
ctypedef np.float64_t [:] double_array_1d_t

ctypedef fused float_or_double_array_1d:
    float_array_1d_t
    double_array_1d_t

ctypedef fused float_or_double:
    float
    double

cdef class AtomBox:
    cdef:
        double [:] periodic_boundaries

    cdef double distance(self, double * atompos_1, double * atompos_2) nogil
    cdef double angle(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil

cdef class Function:
    cdef double evaluate(self, double x)

cdef class AEFunction(Function):
    cdef double A, a, x0, xint, T

cdef class FermiFunction(Function):
    cdef double a, b, c