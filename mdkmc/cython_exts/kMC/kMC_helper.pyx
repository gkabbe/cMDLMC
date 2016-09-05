# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import time
import types
import numpy as np

cimport numpy as np
from cython_gsl cimport *
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport *
cimport mdkmc.cython_exts.atoms.numpyatom as cnpa

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
    """The AtomBox class takes care of all the distance and angle calculation within the kMC_helper."""
    cdef:
        public double[:, :, ::1] trajectory
        double[:] periodic_boundaries
        public double[:] periodic_boundaries_extended
        int[:] box_multiplier
        double[:, ::1] pbc_matrix
        double[:, :, :, :, ::1] frame_reshaped
        int size  # Number of atoms in the extended box
        int atom_number_unextended

    def __cinit__(self, double[:, :, ::1] trajectory,
                  np.ndarray[np.double_t, ndim=1] periodic_boundaries, int[:] box_multiplier):
        self.trajectory = trajectory
        self.periodic_boundaries = periodic_boundaries
        self.box_multiplier = box_multiplier
        self.atom_number_unextended = self.trajectory.shape[1]
        self.size = box_multiplier[0] * box_multiplier[1] * \
            box_multiplier[2] * trajectory.shape[1]

    cdef double distance_ptr(self, double * atompos_1, double * atompos_2) nogil:
        return 0

    cpdef double distance(self, double[:] atompos_1, double[:] atompos_2):
        return 0
        
    cpdef double distance_indices(self, int frame, int index_1, int index_2):
        extended_frame = self.get_extended_frame(frame, self.oxygen_trajectory)
        return self.distance(extended_frame[index_1], extended_frame[index_2])

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return 0
        
    cdef double position_of_index(self, int index, double * position_array):
        cdef int atom_index, box_index, i, j, k
        
        atom_index = index % self.atom_number_unextended
        box_index = index / self.atom_number_unextended

        i = box_index / (self.box_multiplier[1] * self.box_multiplier[2])
        j = (box_index / self.box_multiplier[2]) % self.box_multiplier[1]
        k = box_index % self.box_multiplier[2]

    cdef get_extended_frame_inplace(self, int atomnumber_unextended, double[:, ::1] frame_to_be_extended):
        cdef:
            int x, y, z, i, j

        self.frame_reshaped = <double[:self.box_multiplier[0], :self.box_multiplier[1],
                                      :self.box_multiplier[2], :atomnumber_unextended, :3] > &frame_to_be_extended[0, 0]

        for x in range(self.box_multiplier[0]):
            for y in range(self.box_multiplier[1]):
                for z in range(self.box_multiplier[2]):
                    for i in range(atomnumber_unextended):
                        for j in range(3):
                            if x + y + z != 0:
                                self.frame_reshaped[x, y, z, i, j] = self.frame_reshaped[0, 0, 0, i, j] \
                                    + x * self.pbc_matrix[0, j] \
                                    + y * self.pbc_matrix[1, j] \
                                    + z * self.pbc_matrix[2, j]

    def get_extended_frame(self, double[:, ::1] unextended_frame):
        cdef np.ndarray [np.double_t, ndim = 2] extended_frame  = \
            np.zeros((self.box_multiplier[0] * self.box_multiplier[1] * self.box_multiplier[2] * unextended_frame.shape[0], 3))
        extended_frame[:unextended_frame.shape[0]] = unextended_frame
        self.get_extended_frame_inplace(unextended_frame.shape[0], extended_frame)
        return extended_frame

cdef class AtomBox_Cubic(AtomBox):
    """Subclass of AtomBox, which takes care of orthogonal periodic MD boxes"""

    def __cinit__(self, double[:, :, ::1] oxygen_trajectory,
                  np.ndarray[np.double_t, ndim=1] periodic_boundaries, int[:] box_multiplier,
                  double[:, :, ::1] phosphorus_trajectory=None):
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
        return cnpa.length_ptr( & atompos_1[0], & atompos_2[0], & self.periodic_boundaries_extended[0])

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return cnpa.angle_ptr(atompos_2, atompos_1, atompos_2, atompos_3, & self.periodic_boundaries_extended[0])


cdef class AtomBox_Monoclin(AtomBox):
    """Subclass of AtomBox, which takes care of monoclinic periodic MD boxes"""
    cdef:
        public double[:, ::1] h
        public double[:, ::1] h_inv

    def __cinit__(self, double[:, :, ::1] oxygen_trajectory,
                  np.ndarray[np.double_t, ndim=1] periodic_boundaries, int[:] box_multiplier,
                  double[:, :, ::1] phosphorus_trajectory=None):

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

    cdef double angle_ptr(self, double * atompos_1, double * atompos_2, double * atompos_3) nogil:
        return cnpa.angle_ptr_nonortho(atompos_2, atompos_1, atompos_2, atompos_3, & self.h[0, 0], & self.h_inv[0, 0])

cdef class Helper:
    cdef:
        gsl_rng * r
        int jumps
        # Create containers for the oxygen index from which the proton jump start, for the index of
        # the destination oxygen, and for the jump probability of the oxygen connection.
        # The nested vectors hold
        # the indices and probabilities for the whole trajectory
        public vector[vector[np.int32_t]] start
        public vector[vector[np.int32_t]] destination
        public vector[vector[np.float32_t]] jump_probability
        public vector[vector[int]] neighbors
        int oxygennumber
        int phosphorusnumber
        double r_cut, angle_threshold
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
        self.oxygennumber = atombox.oxygen_trajectory.shape[1]
        self.phosphorusnumber = atombox.phosphorus_trajectory.shape[1]
        self.P_neighbors = P_neighbors
        self.r_cut = cutoff_radius
        self.angle_threshold = angle_threshold
        self.neighbor_search_radius = neighbor_search_radius
        self.saved_frame_counter = 0
        self.calculate_jumpmatrix = calculate_jumpmatrix
        self.jumpmatrix = np.zeros(
            (self.oxygen_frame_extended.shape[0],
             self.oxygen_frame_extended.shape[0]))

        self.atombox = atombox

        if type(seed) != int:
            seed = time.time()
        gsl_rng_set(self.r, seed)
        if verbose:
            print "#Using seed", seed

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
            raise Exception("Jumprate type unknown. Please choose between "
                            "MD_rates and AE_rates")

    def __dealloc__(self):
        gsl_rng_free(self.r)

    def determine_neighbors(self, int framenumber, verbose=False):
        cdef:
            int i, j
            double dist
            vector[int] neighbor_list
        self.oxygen_frame_extended[
            :self.oxygennumber_unextended] = self.atombox.oxygen_trajectory[framenumber]
        self.phosphorus_frame_extended[
            :self.phosphorusnumber_unextended] = self.atombox.phosphorus_trajectory[framenumber]

        self.atombox.get_extended_frame_inplace(
            self.oxygennumber_unextended, self.oxygen_frame_extended)
        self.atombox.get_extended_frame_inplace(
            self.phosphorusnumber_unextended,
            self.phosphorus_frame_extended)

        self.neighbors.clear()
        for i in xrange(self.oxygen_frame_extended.shape[0]):
            neighbor_list.clear()
            for j in xrange(self.oxygen_frame_extended.shape[0]):
                if i != j:
                    dist = self.atombox.distance_ptr(& self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[j, 0])
                    if dist < self.neighbor_search_radius:
                        neighbor_list.push_back(j)
            self.neighbors.push_back(neighbor_list)

    cdef calculate_transitions_POOangle(self, int framenumber, double r_cut, double angle_thresh):
        cdef:
            int i, j, index2
            double dist, PO_angle
            vector[np.int32_t] start_tmp
            vector[np.int32_t] destination_tmp
            vector[np.float32_t] jump_probability_tmp

        self.oxygen_frame_extended[
            :self.oxygennumber_unextended] = self.atombox.oxygen_trajectory[framenumber]
        self.phosphorus_frame_extended[
            :self.phosphorusnumber_unextended] = self.atombox.phosphorus_trajectory[framenumber]

        self.atombox.get_extended_frame_inplace(
            self.oxygennumber_unextended, self.oxygen_frame_extended)
        self.atombox.get_extended_frame_inplace(
            self.phosphorusnumber_unextended,
            self.phosphorus_frame_extended)

        start_tmp.clear()
        destination_tmp.clear()
        jump_probability_tmp.clear()
        for i in range(self.oxygen_frame_extended.shape[0]):
            for j in range(self.neighbors[i].size()):
                index2 = self.neighbors[i][j]
                dist = self.atombox.distance_ptr(& self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[index2, 0])
                if dist < r_cut:
                    POO_angle = self.atombox.angle_ptr( & self.phosphorus_frame_extended[self.P_neighbors[i], 0],
                                                       & self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[index2, 0])
                    start_tmp.push_back(i)
                    destination_tmp.push_back(self.neighbors[i][j])
                    if POO_angle >= angle_thresh:
                        jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))
                    else:
                        jump_probability_tmp.push_back(0)

        self.start.push_back(start_tmp)
        self.destination.push_back(destination_tmp)
        self.jump_probability.push_back(jump_probability_tmp)
        self.saved_frame_counter += 1

    cdef calculate_transitions_POOangle_noneighborlist(self, int framenumber, double r_cut, double angle_thresh):
        cdef:
            int i, j, index2
            double dist, PO_angle
            vector[np.int32_t] start_tmp
            vector[np.int32_t] destination_tmp
            vector[np.float32_t] jump_probability_tmp

        self.oxygen_frame_extended[
            :self.oxygennumber_unextended] = self.atombox.oxygen_trajectory[framenumber]
        self.phosphorus_frame_extended[
            :self.phosphorusnumber_unextended] = self.atombox.phosphorus_trajectory[framenumber]

        self.atombox.get_extended_frame_inplace(
            self.oxygennumber_unextended, self.oxygen_frame_extended)
        self.atombox.get_extended_frame_inplace(
            self.phosphorusnumber_unextended,
            self.phosphorus_frame_extended)

        start_tmp.clear()
        destination_tmp.clear()
        jump_probability_tmp.clear()
        for i in range(self.oxygen_frame_extended.shape[0]):
            for j in range(self.oxygen_frame_extended.shape[0]):
                if i != j:
                    dist = self.atombox.distance_ptr(& self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[j, 0])
                    if dist < r_cut:
                        POO_angle = self.atombox.angle_ptr( & self.phosphorus_frame_extended[self.P_neighbors[i], 0],
                                                           & self.oxygen_frame_extended[i, 0], & self.oxygen_frame_extended[j, 0])
                        start_tmp.push_back(i)
                        destination_tmp.push_back(j)
                        if POO_angle >= angle_thresh:
                            jump_probability_tmp.push_back(self.jumprate_fct.evaluate(dist))
                        else:
                            jump_probability_tmp.push_back(0)

        self.start.push_back(start_tmp)
        self.destination.push_back(destination_tmp)
        self.jump_probability.push_back(jump_probability_tmp)
        self.saved_frame_counter += 1

    def get_transition_number(self):
        return self.start_tmp.size()

    def jumprate_test(self, double x):
        return self.jumprate_fct.evaluate(x)

    def print_transitions(self, int frame):
        cdef int i
        for i in range(self.start[frame].size()):
            print self.start[frame][i], self.destination[frame][i], self.jump_probability[frame][i]
        print "In total", self.start[frame].size(), "connections"

    def return_transitions(self, int frame):
        return self.start[frame], self.destination[frame], self.jump_probability[frame]

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
            self.calculate_transitions_POOangle(i, self.r_cut, self.angle_threshold)
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

        steps = self.start[frame].size()

        for step in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start[frame][i]
            index_destination = self.destination[frame][i]
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
            self.calculate_transitions_POOangle(
                self.saved_frame_counter, self.r_cut, self.angle_threshold)
        steps = self.start[frame].size()
        for step in xrange(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start[frame][i]
            index_destination = self.destination[frame][i]
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
