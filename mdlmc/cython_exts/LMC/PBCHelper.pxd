cdef class AtomBox:
    cdef:
        public double[:, :, ::1] oxygen_trajectory
        public double[:, :, ::1] phosphorus_trajectory
        double[:] periodic_boundaries
        public double[:] periodic_boundaries_extended
        double[:, ::1] pbc_matrix
        int[:] box_multiplier
        int oxygen_number_extended
        int phosphorus_number_extended

    cdef void position_extended_box_ptr(self, int index, double *frame, int frame_len,
                                        double *position) nogil

    cdef double length_extended_box_ptr(self, int index_1, double *frame_1, int frame_1_len,
                                        int index_2, double *frame_2, int frame_2_len) nogil

    cdef double angle_extended_box(self, int index_1, double *frame_1, int frame_1_len,
                                    int index_2, double *frame_2, int frame_2_len, int index_3,
                                    double *frame_3, int frame_3_len) nogil

    cdef double length_(self, double[:] atompos_1, double[:] atompos_2)

    cdef double length_ptr(self, double *atompos_1, double *atompos_2) nogil

    cdef void distance_vector(self, double *atompos_1, double *atompos_2, double *diffvec) nogil

    cpdef double angle(self, double [:] atompos_1, double [:] atompos_2, double [:] atompos_3)

    cdef double angle_ptr(self, double *atompos_1, double *atompos_2, double *atompos_3) nogil

    cdef void distance_vector_extended_box_ptr(self, int index_1, double *frame_1, int frame_1_len,
                                               int index_2, double *frame_2, int frame_2_len,
                                               double *diffvec) nogil


cdef class AtomBoxCubic(AtomBox):
    pass

cdef class AtomBoxMonoclinic(AtomBox):
    cdef:
        public double[:, ::1] h
        public double[:, ::1] h_inv
