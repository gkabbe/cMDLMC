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

# cdef double fermi(double a, double b, double c, double x) nogil