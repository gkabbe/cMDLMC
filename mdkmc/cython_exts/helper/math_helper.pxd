cdef void matrix_mult(double [:,::1] mat, double [:] vec) nogil
cdef void matrix_mult_ptr(double * mat, double * vec) nogil
cdef double dot_product(double* a1, double* a2, int length) nogil
