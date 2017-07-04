# cython: profile = False
# cython: boundscheck = False, wraparound = False, cdivision = True, initializedcheck = False
# cython: language_level = 3

import cython
cimport cython
import numpy as np
import logging

cimport numpy as np

cimport mdlmc.cython_exts.atoms.numpyatom as cnpa
cimport mdlmc.cython_exts.helper.math_helper as mh

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


cdef class AtomBox:
    """The AtomBox class takes care of all distance and angle calculations.
    Depending on the periodic boundary conditions of the system, either the subclass
    AtomBoxCubic or AtomBoxMonoclinic need to be instantiated."""
    # cdef:
    #     public double[:, :, ::1] oxygen_trajectory
    #     public double[:, :, ::1] phosphorus_trajectory
    #     double[:] periodic_boundaries
    #     public double[:] periodic_boundaries_extended
    #     double[:, ::1] pbc_matrix
    #     int[:] box_multiplier
    #     int oxygen_number_extended
    #     int phosphorus_number_extended

    def __cinit__(self, periodic_boundaries, *args, box_multiplier=(1, 1, 1), **kwargs):
        self.periodic_boundaries = np.array(periodic_boundaries, dtype=float)
        self.box_multiplier = np.array(box_multiplier, dtype=np.int32)

    def position_extended_box(self, int index, double [:, ::1] frame):
        cdef np.ndarray[np.float_t, ndim=1] pos = np.zeros(3)
        self.position_extended_box_ptr(index, &frame[0, 0], frame.shape[0], &pos[0])
        return pos

    cdef void position_extended_box_ptr(self, int index, double *frame, int frame_len,
                                        double *position) nogil:
       cdef:
            int atom_index, box_index, i, j, k, ix

       atom_index = index % frame_len
       box_index = index / frame_len
       i = box_index / (self.box_multiplier[1] * self.box_multiplier[2])
       j = (box_index / self.box_multiplier[2]) % self.box_multiplier[1]
       k = box_index % self.box_multiplier[2]

       for ix in range(3):
           position[ix] =  frame[3 * atom_index + ix] + i * self.pbc_matrix[0, ix] \
                                                      + j * self.pbc_matrix[1, ix] \
                                                      + k * self.pbc_matrix[2, ix]

    @cython.boundscheck(True)
    def distance(self, arr1, arr2):
        """Calculates for two arrays of positions an array of vector distances"""
        cdef:
            int i, j
        arr1 = arr1.reshape((-1, 3)).astype(float)
        arr2 = arr2.reshape((-1, 3)).astype(float)

        cdef:
            double [:, ::1] arr1_view = arr1
            double [:, ::1] arr2_view = arr2
            np.ndarray[np.float64_t, ndim=2] result = np.zeros(arr1.shape)

        for i in range(arr1_view.shape[0]):
            self.distance_vector(&arr1_view[i, 0], &arr2_view[i, 0], &result[i, 0])
        return np.squeeze(result)

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def length(self, arr1, arr2):
        """Calculates for two arrays of positions an array of scalar distances"""
        cdef:
            int i
            double [:, ::1] arr1_view = arr1.reshape((-1, 3)).astype(float)
            double [:, ::1] arr2_view = arr2.reshape((-1, 3)).astype(float)
        result = np.zeros(arr1_view.shape[0])

        for i in range(arr1_view.shape[0]):
            result[i] = self.length_(arr1_view[i], arr2_view[i])

        return result

    @cython.boundscheck(False)
    def length_all_to_all(self, double [:, ::1] arr1, double [:, ::1] arr2):
        cdef int i, j
        result = np.zeros((arr1.shape[0], arr2.shape[0]))

        for i in range(arr1.shape[0]):
            for j in range(arr2.shape[0]):
                result[i, j] = self.length_ptr(&arr1[i, 0], &arr2[j, 0])
        return result

    cdef double length_extended_box_ptr(self, int index_1, double * frame_1, int frame_1_len,
                                          int index_2, double * frame_2, int frame_2_len) nogil:
        """Calculates the distance between two atoms, taking into account the periodic boundary
        conditions of the extended periodic box."""
        cdef double dist[3]
        self.distance_vector_extended_box_ptr(index_1, frame_1, frame_1_len, index_2, frame_2,
                                              frame_2_len, dist)
        return sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])

    cdef double angle_extended_box(self,
                                   int index_1, double *frame_1, int frame_1_len,
                                   int index_2, double *frame_2, int frame_2_len,
                                   int index_3, double *frame_3, int frame_3_len) nogil:
        """Calculates the angle ∠ index_1 index_2 index_3"""
        cdef:
            double diff_2_1[3]
            double diff_2_3[3]

        self.distance_vector_extended_box_ptr(index_2, frame_2, frame_2_len, index_1, frame_1,
                                              frame_1_len, diff_2_1)
        self.distance_vector_extended_box_ptr(index_2, frame_2, frame_2_len, index_3, frame_3,
                                              frame_3_len, diff_2_3)

        return acos(mh.dot_product_ptr(diff_2_1, diff_2_3, 3) / sqrt(
            mh.dot_product_ptr(diff_2_1, diff_2_1, 3)) / sqrt(
            mh.dot_product_ptr(diff_2_3, diff_2_3, 3)))

    cdef double length_(self, double[:] atompos_1, double[:] atompos_2):
        return 0

    cdef double length_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return 0

    cdef void distance_vector(self, double * atompos_1, double * atompos_2, double * diffvec) nogil:
        pass

    cpdef double angle(self, double [:] atompos_1, double [:] atompos_2, double [:] atompos_3):
        return self.angle_ptr(&atompos_1[0], &atompos_2[0], &atompos_3[0])

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return 0

    cdef void distance_vector_extended_box_ptr(self, int index_1, double * frame_1, int frame_1_len,
                                               int index_2, double * frame_2, int frame_2_len,
                                               double * diffvec) nogil:
        cdef:
            int atom_index, box_index, i, j, k, ix
            double[3] pos_1, pos_2

        if self.box_multiplier[0] == self.box_multiplier[1] == self.box_multiplier[2] == 1:
            self.distance_vector(&frame_1[3 * index_1], &frame_2[3 * index_2], diffvec)
        else:
            self.position_extended_box_ptr(index_1, frame_1, frame_1_len, &pos_1[0])
            self.position_extended_box_ptr(index_2, frame_2, frame_2_len, &pos_2[0])
            self.distance_vector(pos_1, pos_2, diffvec)

    def next_neighbor(self, double [:] pos, double [:, ::1] frame_2):
        cdef:
            double length, minimum_length = 1e30
            int minimum_index = -1
            int index_2
            int atom_number = frame_2.shape[0] * self.box_multiplier[0] * self.box_multiplier[1] * \
                              self.box_multiplier[2]

        for index_2 in range(atom_number):
            length = self.length_ptr(&pos[0], &frame_2[index_2, 0])
            if length < minimum_length:
                minimum_length = length
                minimum_index = index_2

        return minimum_index, minimum_length

    def next_neighbor_extended_box(self, int index_1, double [:, ::1] frame_1,
                                   double [:,::1] frame_2):
        cdef:
            double distance, minimum_distance = 1e30
            int minimum_index = -1
            int index_2
            int atom_number = frame_2.shape[0] * self.box_multiplier[0] * self.box_multiplier[1] * \
                              self.box_multiplier[2]

        for index_2 in range(atom_number):
            distance = self.length_extended_box_ptr(index_1, &frame_1[0, 0], frame_1.shape[0],
                                                  index_2, &frame_2[0, 0], frame_2.shape[0])
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_index = index_2

        return minimum_index, minimum_distance

    def determine_phosphorus_oxygen_pairs(self, double [:, ::1] oxygen_atoms,
                                          double [:, ::1] phosphorus_atoms):
        cdef int oxygen_number_extended = oxygen_atoms.shape[0] * self.box_multiplier[0] * \
                                          self.box_multiplier[1] * self.box_multiplier[2]
        phosphorus_neighbors = np.zeros(oxygen_number_extended, np.int32)
        for oxygen_index in range(oxygen_number_extended):
            phosphorus_index, _ = self.next_neighbor_extended_box(oxygen_index, oxygen_atoms,
                                                                  phosphorus_atoms)
            phosphorus_neighbors[oxygen_index] = phosphorus_index
        return phosphorus_neighbors

    def get_acidic_proton_indices(self, atoms, verbose=False):
        """Expects numpy array 'atoms' of dtype 'xyz_dtype'"""
        acidic_indices = []
        protons = atoms[atoms["name"] == "H"]
        proton_indices, = np.where(atoms["name"] == "H")
        all_other_atoms = atoms[atoms["name"] != "H"]
        for i, single_proton in enumerate(protons):
            nn_index, next_neighbor = self.next_neighbor(single_proton["pos"], all_other_atoms["pos"])
            if all_other_atoms["name"][nn_index] == "O":
                acidic_indices.append(proton_indices[i])
        if verbose:
            print("# Acidic indices: ", acidic_indices)
            print("# Number of acidic protons: ", len(acidic_indices))
        return acidic_indices

cdef class AtomBoxCubic(AtomBox):
    """Subclass of AtomBox for orthogonal periodic MD boxes"""

    def __cinit__(self, periodic_boundaries, *args, box_multiplier=(1, 1, 1), **kwargs):
        cdef int i

        self.pbc_matrix = np.zeros((3, 3))
        self.pbc_matrix[0, 0] = periodic_boundaries[0]
        self.pbc_matrix[1, 1] = periodic_boundaries[1]
        self.pbc_matrix[2, 2] = periodic_boundaries[2]

        self.periodic_boundaries_extended = np.array(periodic_boundaries, dtype=float)
        for i in range(3):
            self.periodic_boundaries_extended[i] *= box_multiplier[i]

    cdef double length_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return cnpa.length_ptr(atompos_1, atompos_2, & self.periodic_boundaries_extended[0])

    cdef double length_(self, double[:] atompos_1, double[:] atompos_2):
        return cnpa.length(atompos_1, atompos_2, self.periodic_boundaries_extended)

    cdef void distance_vector(self, double * atompos_1, double * atompos_2, double * diffvec) nogil:
        cnpa.diff_ptr(atompos_1, atompos_2, &self.periodic_boundaries_extended[0], diffvec)

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return cnpa.angle_ptr(atompos_2, atompos_1, atompos_2, atompos_3,
                              &self.periodic_boundaries_extended[0])


cdef class AtomBoxMonoclinic(AtomBox):
    """Subclass of AtomBox for nonorthogonal periodic MD boxes"""
    # cdef:
    #     public double[:, ::1] h
    #     public double[:, ::1] h_inv

    def __cinit__(self, periodic_boundaries, *args, box_multiplier=(1, 1, 1), **kwargs):

        self.periodic_boundaries_extended = np.array(periodic_boundaries, dtype=float)
        for i in range(0, 3):
            for j in range(0, 3):
                self.periodic_boundaries_extended[3 * i + j] *= box_multiplier[i]

        self.h = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.h[j, i] = self.periodic_boundaries_extended[i * 3 + j]
        self.h_inv = np.array(np.linalg.inv(self.h), order="C")
        self.pbc_matrix = periodic_boundaries.reshape((3, 3))

    cdef double length_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return cnpa.length_nonortho_bruteforce_ptr(atompos_1, atompos_2, &self.h[0, 0],
                                                   &self.h_inv[0, 0])

    cdef double length_(self, double[:] atompos_1, double[:] atompos_2):
        return cnpa.length_nonortho_bruteforce_ptr(&atompos_1[0], &atompos_2[0], &self.h[0, 0],
                                                   &self.h_inv[0, 0])

    cdef void distance_vector(self, double * atompos_1, double * atompos_2, double * diffvec) nogil:
        cnpa.diff_ptr_nonortho(atompos_1, atompos_2, diffvec, &self.h[0, 0], &self.h_inv[0, 0])

    cdef double angle_ptr(self, double *atompos_1, double *atompos_2, double *atompos_3) nogil:
        return cnpa.angle_ptr_nonortho(atompos_2, atompos_1, atompos_2, atompos_3, &self.h[0, 0],
                                       &self.h_inv[0, 0])


cdef class AtomBoxWater(AtomBoxCubic):
    """Converts oxygen-oxygen distances to typical hydronium-oxygen distances"""
    cdef double left_bound, right_bound

    cdef double convert_distance(self, double distance) nogil:
        return distance

    cdef double length_extended_box_ptr(self, int index_1, double * frame_1, int frame_1_len,
                                          int index_2, double * frame_2, int frame_2_len) nogil:
        # Call the method of the super class
        cdef double length = AtomBoxCubic.length_extended_box_ptr(self,
                                                                  index_1, frame_1, frame_1_len,
                                                                  index_2, frame_2, frame_2_len)
        return self.convert_distance(length)

    def length(self, arr1, arr2):
        length = AtomBoxCubic.length(self, arr1, arr2)
        cdef int i
        for i in range(length.shape[0]):
            length[i] = self.convert_distance(length[i])

        return length

    cdef double length_ptr(self, double * arr1, double * arr2) nogil:
        cdef double dist = AtomBoxCubic.length_ptr(self, arr1, arr2)
        return self.convert_distance(dist)


cdef class AtomBoxWaterLinearConversion(AtomBoxWater):
    """Converts oxygen-oxygen distances to typical hydronium-oxygen distances"""
    cdef double a, b

    def __cinit__(self, periodic_boundaries, *args, box_multiplier=(1, 1, 1), **kwargs):
        param_dict = args[0]
        # Parameters for the conversion function
        self.a = param_dict["a"]
        self.b = param_dict["b"]
        self.left_bound = param_dict["left_bound"]
        self.right_bound = param_dict["right_bound"]

    cdef double convert_distance(self, double distance) nogil:
        cdef double new_distance
        if self.left_bound < distance < self.right_bound:
            new_distance = self.a * distance + self.b
        else:
            new_distance = distance
        return new_distance


cdef class AtomBoxWaterRampConversion(AtomBoxWater):
    """Converts oxygen-oxygen distances to typical hydronium-oxygen distances"""

    cdef:
        double a, b, d0

    def __cinit__(self, periodic_boundaries, *args, box_multiplier=(1, 1, 1), **kwargs):
        param_dict = args[0]
        # Parameters for the conversion function
        self.a = param_dict["a"]
        self.b = param_dict["b"]
        self.d0 = param_dict["d0"]
        self.left_bound = param_dict["left_bound"]
        self.right_bound = param_dict["right_bound"]

    cdef double convert_distance(self, double distance) nogil:
        cdef double new_distance
        if self.left_bound < distance < self.right_bound:
            if distance < self.d0:
                new_distance = self.b
            else:
                new_distance = self.a * (distance - self.d0) + self.b
        else:
            new_distance = distance
        return new_distance

