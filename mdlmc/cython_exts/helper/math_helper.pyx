# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# cython: language_level = 3
import cython
cimport numpy as cnp

cdef extern from "math.h":
	double sqrt(double x) nogil
	double exp(double x) nogil
	double cos(double x) nogil
	double acos(double x) nogil
	
#~ cdef extern from "<cmath>":
#~ 	double round(double x) nogil

cdef double dot_product_ptr(double* a1, double* a2, int length) nogil:
	cdef:
		int i
		double result = 0

	for i in range(length):
		result += a1[i] * a2[i]
	return result

cdef double dot_product(double [:] a1, double [:] a2) nogil:
	cdef:
		int i
		double result = 0

	for i in range(a1.shape[0]):
		result += a1[i] * a2[i]
	return result

#only valid for 3d!!
cdef void matrix_mult(double [:,::1] mat, double [:] vec) nogil:
	cdef:
		int i, j
		double * resultvec = [0,0,0]

	for i in range(3):
		for j in range(3):
			resultvec[i] += mat[i,j] * vec[j]

	for i in range(3):
		vec[i] = resultvec[i]

def round_test(double x):
	return round(x)

cdef void matrix_mult_ptr(double * mat, double * vec) nogil:
	cdef:
		int i, j
		double * resultvec = [0,0,0]

	for i in range(3):
		for j in range(3):
			resultvec[i] += mat[3*i+j] * vec[j]
	
	for i in range(3):
		vec[i] = resultvec[i]

cdef read_mat(double * mat):
	cdef:
		int i,j
		
	for i in range(3):
		for j in range(3):
			print(mat[3*i+j])
		
def mat_test(double [:,::1] mat, double[:] vec):
	matrix_mult_ptr(&mat[0,0], &vec[0])
#~ 	print vec[0], vec[1], vec[2]
#~ 	return vec
	
def test(double [:,::1] mat):
	read_mat(&mat[0,0])
