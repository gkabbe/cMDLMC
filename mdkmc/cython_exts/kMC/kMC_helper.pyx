# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import time
import numpy as np

cimport numpy as np
from cython_gsl cimport *
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport *
cimport mdkmc.cython_exts.atoms.numpyatom as cnpa
cimport mdkmc.cython_exts.helper.math_helper as mh

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil

ctypedef np.int_t DTYPE_t
cdef double PI = np.pi

cdef double R = 1.9872041e-3   # universal gas constant in kcal/mol/K


# Define Function Objects. These can later be substituted easily by kMC_helper
cdef class JumprateFunction:
    cpdef double evaluate(self, double x):
        return 0


cdef class FermiFunction(JumprateFunction):
    cdef:
        double a, b, c

    def __cinit__(self, double a, double b, double c):
        self.a = a
        self.b = b
        self.c = c

    cpdef double evaluate(self, double x):
        return self.a / (1 + exp((x - self.b) / self.c))


cdef class AEFunction(JumprateFunction):
    cdef:
        double A, a, b, d0, T

    def __cinit__(self, double A, double a, double b, double d0, double T):
        self.A = A
        self.a = a
        self.b = b
        self.d0 = d0
        self.T = T

    cpdef double evaluate(self, double x):
        cdef double E
        if x <= self.d0:
            E = 0
        else:
            E = self.a * (x - self.d0) / sqrt(self.b + 1.0 / (x - self.d0) / (x - self.d0))
        return self.A * exp(-E / (R * self.T))

    cpdef double get_energy(self, double x):
        if x <= self.x0:
            return 0
        else:
            return self.a * (x - self.d0) / sqrt(self.b + 1.0 / (x - self.d0) / (x - self.d0))


cdef class AtomBox:
    """The AtomBox class takes care of all distance and angle calculations."""
    cdef:
        public double[:, :, ::1] oxygen_trajectory
        public double[:, :, ::1] phosphorus_trajectory
        public int[:] phosphorus_neighbors
        double[:] periodic_boundaries
        public double[:] periodic_boundaries_extended
        double[:, ::1] pbc_matrix
        int[:] box_multiplier
        int oxygen_number_extended
        int phosphorus_number_extended

    def __cinit__(self, double[:, :, ::1] oxygen_trajectory, double[:, :, ::1] phosphorus_trajectory,
                  periodic_boundaries, box_multiplier):
        self.oxygen_trajectory = oxygen_trajectory
        self.phosphorus_trajectory = phosphorus_trajectory
        self.periodic_boundaries = np.asfarray(periodic_boundaries)
        self.box_multiplier = np.asarray(box_multiplier, dtype=np.int32)
        self.oxygen_number_extended = self.oxygen_trajectory.shape[1] * box_multiplier[0] * \
                                      box_multiplier[1] * box_multiplier[2]

    def __init__(self, *args, **kwargs):
        self.phosphorus_neighbors = self.determine_phosphorus_oxygen_pairs(0)

    cpdef double distance_extended_box(self, int index_1, double[:, ::1] frame_1, int index_2,
                                       double[:, ::1] frame_2):
        """Calculates the distance between two atoms, taking into account the periodic boundary
        conditions of the extended periodic box."""
        cdef double[:] dist = self.distance_vector_extended_box(index_1, frame_1, index_2, frame_2)
        return sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])

    cpdef double angle_extended_box(self, int index_1, double[:, ::1] frame_1, int index_2,
                                    double[:, ::1] frame_2, int index_3, double[:, ::1] frame_3):
        """Calculates the angle ∠ index_1 index_2 index_3"""
        cdef double[:] diff_2_1, diff_2_3

        diff_2_1 = self.distance_vector_extended_box(index_2, frame_2, index_1, frame_1)
        diff_2_3 = self.distance_vector_extended_box(index_2, frame_2, index_3, frame_3)

        return acos(mh.dot_product(diff_2_1, diff_2_3) / sqrt(
            mh.dot_product(diff_2_1, diff_2_1)) / sqrt(mh.dot_product(diff_2_3, diff_2_3)))

    cpdef double[:] distance_vector(self, double[:] atompos_1, double[:] atompos_2):
        cdef double x[3]
        return x

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return 0

    cpdef double[:] distance_vector_extended_box(self, int index_1, double[:, ::1] frame_1,
                                      int index_2, double[:, ::1] frame_2):
        cdef: 
            int atom_index, box_index, i, j, k, ix
            double[3] pos_1, pos_2, distance
        
        atom_index = index_1 % frame_1.shape[0]
        box_index = index_1 / frame_1.shape[0]
        
        i = box_index / (self.box_multiplier[1] * self.box_multiplier[2])
        j = (box_index / self.box_multiplier[2]) % self.box_multiplier[1]
        k = box_index % self.box_multiplier[2]
        
        for ix in range(3):
            pos_1[ix] =  frame_1[atom_index, ix] + i * self.pbc_matrix[0, ix] \
                                                 + j * self.pbc_matrix[1, ix] \
                                                 + k * self.pbc_matrix[2, ix]

        atom_index = index_2 % frame_2.shape[0]
        box_index = index_2 / frame_2.shape[0]

        i = box_index / (self.box_multiplier[1] * self.box_multiplier[2])
        j = (box_index / self.box_multiplier[2]) % self.box_multiplier[1]
        k = box_index % self.box_multiplier[2]
        
        for ix in range(3):
            pos_2[ix] =  frame_2[atom_index, ix] + i * self.pbc_matrix[0, ix] \
                                                 + j * self.pbc_matrix[1, ix] \
                                                 + k * self.pbc_matrix[2, ix]
                                              
        return self.distance_vector(pos_1, pos_2)

    def next_neighbor(self, int index_1, double [:, ::1] frame_1, double [:, ::1] frame_2):
        cdef:
            double distance, minimum_distance = 1e30
            int minimum_index = -1
            int index_2
            int atom_number = frame_2.shape[0] * self.box_multiplier[0] * self.box_multiplier[1] * \
                              self.box_multiplier[2]

        for index_2 in range(atom_number):
            distance = self.distance_extended_box(index_1, frame_1, index_2, frame_2)
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_index = index_2

        return minimum_index, minimum_distance

    def determine_phosphorus_oxygen_pairs(self, frame_number):
        phosphorus_neighbors = np.zeros(self.oxygen_number_extended, np.int32)
        oxygen_atoms = self.oxygen_trajectory[frame_number]
        phosphorus_atoms = self.phosphorus_trajectory[frame_number]

        for oxygen_index in range(self.oxygen_number_extended):
            phosphorus_index, _ = self.next_neighbor(oxygen_index, oxygen_atoms, phosphorus_atoms)
            phosphorus_neighbors[oxygen_index] = phosphorus_index
        return phosphorus_neighbors


cdef class AtomBoxCubic(AtomBox):
    """Subclass of AtomBox for orthogonal periodic MD boxes"""

    def __cinit__(self, double[:, :, ::1] oxygen_trajectory, 
                  double[:, :, ::1] phosphorus_trajectory,
                  np.ndarray[np.double_t, ndim=1] periodic_boundaries, box_multiplier):
        cdef int i

        self.pbc_matrix = np.zeros((3, 3))
        self.pbc_matrix[0, 0] = periodic_boundaries[0]
        self.pbc_matrix[1, 1] = periodic_boundaries[1]
        self.pbc_matrix[2, 2] = periodic_boundaries[2]

        self.periodic_boundaries_extended = np.array(periodic_boundaries)
        for i in range(3):
            self.periodic_boundaries_extended[i] *= box_multiplier[i]

    cdef double distance_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return cnpa.length_ptr(atompos_1, atompos_2, & self.periodic_boundaries_extended[0])

    cpdef double distance(self, double[:] atompos_1, double[:] atompos_2):
        return cnpa.length(atompos_1, atompos_2, self.periodic_boundaries_extended)

    cpdef double[:] distance_vector(self, double[:] atompos_1, double[:] atompos_2):
        cdef double x[3]
        cnpa.diff_ptr(&atompos_1[0], &atompos_2[0], &self.periodic_boundaries_extended[0], &x[0])
        return x

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return cnpa.angle_ptr(atompos_2, atompos_1, atompos_2, atompos_3, & self.periodic_boundaries_extended[0])

    cpdef int next_neighbor(self, int index_1, double [:, ::1] frame_1, double [:, ::1] frame_2):

cdef class AtomBoxMonoclin(AtomBox):
    """Subclass of AtomBox for monoclinic periodic MD boxes"""
    cdef:
        public double[:, ::1] h
        public double[:, ::1] h_inv

    def __cinit__(self, double[:, :, ::1] oxygen_trajectory, 
                  double[:, :, ::1] phosphorus_trajectory,
                  np.ndarray[np.double_t, ndim=1] periodic_boundaries, box_multiplier):

        self.periodic_boundaries_extended = np.array(periodic_boundaries)
        for i in range(0, 3):
            for j in range(0, 3):
                self.periodic_boundaries_extended[3 * i + j] *= box_multiplier[i]

        self.h = np.zeros((3, 3))
        for i in xrange(3):
            for j in xrange(3):
                self.h[j, i] = self.periodic_boundaries_extended[i * 3 + j]
        self.h_inv = np.array(np.linalg.inv(self.h), order="C")
        self.pbc_matrix = periodic_boundaries.reshape((3, 3))

    cdef double distance_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return cnpa.length_nonortho_bruteforce_ptr(atompos_1, atompos_2, & self.h[0, 0], & self.h_inv[0, 0])

    cpdef double distance(self, double[:] atompos_1, double[:] atompos_2):
        return cnpa.length_nonortho_bruteforce_ptr(& atompos_1[0], & atompos_2[0], & self.h[0, 0], & self.h_inv[0, 0])

    cpdef double[:] distance_vector(self, double[:] atompos_1, double[:] atompos_2):
        cdef double x[3]
        cnpa.diff_nonortho(atompos_1, atompos_2, x, self.h, self.h_inv)
        return x

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return cnpa.angle_ptr_nonortho(atompos_2, atompos_1, atompos_2, atompos_3, & self.h[0, 0], & self.h_inv[0, 0])
        
    def determine_phosphorus_oxygen_pairs(self, frame_number):
        phosphorus_neighbors = np.zeros(self.oxygennumber_extended, int)
        oxygen_atoms = self.oxygen_trajectory[frame_number]
        phosphorus_atoms = self.phosphorus_trajectory[frame_number]

        for i in range(oxygen_atoms.shape[0]):
            phosphorus_index = \
                cnpa.next_neighbor_nonortho(oxygen_atoms[i], phosphorus_atoms, self.h, 
                                            self.h_inv)[0]
            phosphorus_neighbors[i] = phosphorus_index
        return phosphorus_neighbors


cdef class LMCRoutine:
    """Main component of the cMD/LMC algorithm.
    This class determines the jump rates within a molecule according to the atomic distances and
    carries out the Monte Carlo steps."""

    cdef:
        gsl_rng * r
        int jumps
        # Create containers for the oxygen index from which the proton jump start, for the index of
        # the destination oxygen, and for the jump probability of the oxygen connection.
        # The nested vectors hold the indices and probabilities for the whole trajectory
        public vector[vector[np.int32_t]] start_indices
        public vector[vector[np.int32_t]] destination_indices
        public vector[vector[np.float32_t]] jump_probability
        public vector[vector[int]] neighbors
        int oxygen_number
        int phosphorus_number
        double cutoff_radius, angle_threshold
        double neighbor_search_radius
        public JumprateFunction jumprate_fct
        public AtomBox atombox
        np.uint32_t saved_frame_counter
        public double[:, ::1] jumpmatrix
        bool calculate_jumpmatrix

    def __cinit__(self, AtomBox atombox, jumprate_parameter_dict, double cutoff_radius,
                  double angle_threshold, double neighbor_search_radius, jumprate_type, *,
                  seed=None, bool calculate_jumpmatrix=False, verbose=False):
        cdef:
            int i
            double[:] pbc_extended
        self.r = gsl_rng_alloc(gsl_rng_mt19937)
        self.jumps = 0
        self.oxygen_number = atombox.oxygen_number_extended
        self.phosphorus_number = atombox.phosphorus_number_extended
        self.cutoff_radius = cutoff_radius
        self.angle_threshold = angle_threshold
        self.neighbor_search_radius = neighbor_search_radius
        self.saved_frame_counter = 0
        self.calculate_jumpmatrix = calculate_jumpmatrix
        self.jumpmatrix = np.zeros((self.oxygen_number, self.oxygen_number))
        self.atombox = atombox
        if type(seed) != int:
            seed = time.time()
        gsl_rng_set(self.r, seed)
        if verbose:
            print "# Using seed", seed
        if jumprate_type == "MD_rates":
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            c = jumprate_parameter_dict["c"]
            self.jumprate_fct = FermiFunction(a, b, c)
        elif jumprate_type == "AE_rates":
            A = jumprate_parameter_dict["A"]
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            d0 = jumprate_parameter_dict["d0"]
            T = jumprate_parameter_dict["T"]
            self.jumprate_fct = AEFunction(A, a, b, d0, T)
        else:
            raise Exception("Jump rate type unknown. Please choose between "
                            "MD_rates and AE_rates")

    def __dealloc__(self):
        gsl_rng_free(self.r)

    def determine_neighbors(self, int frame_number, verbose=False):
        cdef:
            int i, j
            double dist
            vector[int] neighbor_list

        self.neighbors.clear()
        for i in xrange(self.atombox.oxygen_number_extended):
            neighbor_list.clear()
            for j in xrange(self.atombox.oxygen_number_extended):
                if i != j:
                    dist = self.atombox.distance_extended_box(i, self.atombox.oxygen_trajectory[
                        frame_number], j, self.atombox.oxygen_trajectory[frame_number])
                    if dist < self.neighbor_search_radius:
                        neighbor_list.push_back(j)
            self.neighbors.push_back(neighbor_list)

    cdef calculate_transitions(self, int frame_number, double r_cut, double angle_thresh):
        cdef:
            int start_index, neighbor_index, destination_index
            double dist, PO_angle
            vector[np.int32_t] start_indices_tmp
            vector[np.int32_t] destination_indices_tmp
            vector[np.float32_t] jump_probability_tmp

        start_indices_tmp.clear()
        destination_indices_tmp.clear()
        jump_probability_tmp.clear()
        for start_index in range(self.atombox.oxygen_number_extended):
            for neighbor_index in range(self.neighbors[start_index].size()):
                destination_index = self.neighbors[start_index][neighbor_index]
                dist = self.atombox.distance_extended_box(start_index,
                                                      self.atombox.oxygen_trajectory[frame_number],
                                                      destination_index,
                                                      self.atombox.oxygen_trajectory[frame_number])
                if dist < r_cut:
                    poo_angle = self.atombox.angle_extended_box(
                        self.atombox.phosphorus_neighbors[start_index],
                        self.atombox.phosphorus_trajectory[frame_number],
                        start_index, self.atombox.oxygen_trajectory[frame_number],
                        destination_index, self.atombox.oxygen_trajectory[frame_number])
                    start_indices_tmp.push_back(start_index)
                    destination_indices_tmp.push_back(destination_index)
                    if poo_angle >= angle_thresh:
                        jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))
                    else:
                        jump_probability_tmp.push_back(0)

        self.start_indices.push_back(start_indices_tmp)
        self.destination_indices.push_back(destination_indices_tmp)
        self.jump_probability.push_back(jump_probability_tmp)
        self.saved_frame_counter += 1

    # cdef calculate_transitions_POOangle_noneighborlist(self, int framenumber, double r_cut, double angle_thresh):
    #     cdef:
    #         int i, j, index2
    #         double dist, PO_angle
    #         vector[np.int32_t] start_tmp
    #         vector[np.int32_t] destination_tmp
    #         vector[np.float32_t] jump_probability_tmp
    #
    #     self.oxygen_frame_extended[
    #         :self.oxygennumber_unextended] = self.atombox.oxygen_trajectory[framenumber]
    #     self.phosphorus_frame_extended[
    #         :self.phosphorusnumber_unextended] = self.atombox.phosphorus_trajectory[framenumber]
    #
    #     self.atombox.get_extended_frame_inplace(
    #         self.oxygennumber_unextended, self.oxygen_frame_extended)
    #     self.atombox.get_extended_frame_inplace(
    #         self.phosphorusnumber_unextended,
    #         self.phosphorus_frame_extended)
    #
    #     start_tmp.clear()
    #     destination_tmp.clear()
    #     jump_probability_tmp.clear()
    #     for i in range(self.oxygen_frame_extended.shape[0]):
    #         for j in range(self.oxygen_frame_extended.shape[0]):
    #             if i != j:
    #                 dist = self.atombox.distance_ptr(& self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[j, 0])
    #                 if dist < r_cut:
    #                     POO_angle = self.atombox.angle_ptr( & self.phosphorus_frame_extended[self.P_neighbors[i], 0],
    #                                                        & self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[j, 0])
    #                     start_tmp.push_back(i)
    #                     destination_tmp.push_back(j)
    #                     if POO_angle >= angle_thresh:
    #                         jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))
    #                     else:
    #                         jump_probability_tmp.push_back(0)
    #
    #     self.start_indices.push_back(start_tmp)
    #     self.destination_indices.push_back(destination_tmp)
    #     self.jump_probability.push_back(jump_probability_tmp)
    #     self.saved_frame_counter += 1

    def get_transition_number(self):
        return self.start_tmp.size()

    def jumprate_test(self, double x):
        return self.jumprate_fct.evaluate(x)

    def print_transitions(self, int frame):
        cdef int i
        for i in range(self.start_indices[frame].size()):
            print self.start_indices[frame][i], self.destination_indices[frame][i], \
                  self.jump_probability[frame][i]
        print "In total", self.start_indices[frame].size(), "connections"

    def return_transitions(self, int frame):
        return self.start_indices[frame], self.destination_indices[frame], self.jump_probability[frame]

    def sweep_list(self, np.uint8_t[:] proton_lattice):
        cdef:
            int i, j, index_origin, index_destin
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

    def store_transitions_in_vector(self, bool verbose=False):
        cdef int i
        for i in range(self.atombox.oxygen_trajectory.shape[0]):
            self.calculate_transitions(i, self.cutoff_radius, self.angle_threshold)
            if verbose and i % 1000 == 0:
                print "# Saving transitions {} / {}".format(i, self.atombox.oxygen_trajectory.shape[0]), "\r",
        print ""
        if verbose:
            print "# Done"

    def sweep_from_vector(self, int frame, np.uint8_t[:] proton_lattice):
        cdef:
            int step, i, index_origin, index_destination
            int trajectory_length
            int steps

        steps = self.start_indices[frame].size()

        for step in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start_indices[frame][i]
            index_destination = self.destination_indices[frame][i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destination] == 0:
                if gsl_rng_uniform(self.r) < self.jump_probability[frame][i]:
                    proton_lattice[index_destination] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    self.jumps += 1

    def sweep_from_vector_jumpmatrix(self, int frame, np.uint8_t[:] proton_lattice):
        cdef:
            int step, i, index_origin, index_destination
            int trajectory_length
            int steps
        trajectory_length = self.atombox.oxygen_trajectory.shape[0]
        while self.saved_frame_counter < trajectory_length and self.saved_frame_counter < frame + 1:
            self.calculate_transitions(
                self.saved_frame_counter, self.cutoff_radius, self.angle_threshold)
        steps = self.start_indices[frame].size()
        for step in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start_indices[frame][i]
            index_destination = self.destination_indices[frame][i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destination] == 0:
                if gsl_rng_uniform(self.r) < self.jump_probability[frame][i]:
                    proton_lattice[index_destination] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    self.jumps += 1
                    self.jumpmatrix[index_origin, index_destination] += 1

    def reset_jumpcounter(self):
        self.jumps = 0

    def get_jumps(self):
        return self.jumps

