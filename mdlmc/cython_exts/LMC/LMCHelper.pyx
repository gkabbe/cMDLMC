# cython: profile = False
# cython: boundscheck = False, wraparound = False, cdivision = True, initializedcheck = False
# cython: language_level = 3
import time

import numpy as np
import cython
cimport numpy as np
from cython_gsl cimport *
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport *

from mdlmc.cython_exts.LMC.PBCHelper cimport AtomBox, AtomBoxCubic, AtomBoxMonoclinic

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double acos(double x) nogil

ctypedef np.int_t DTYPE_t
cdef double PI = np.pi

cdef double R = 1.9872041e-3   # universal gas constant in kcal/mol/K


# Define Function Objects. These can later be substituted easily by LMCRoutine
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


cdef class ExponentialFunction(JumprateFunction):
    cdef:
        double a, b
        
    def __cinit__(self, double a, double b):
        self.a = a
        self.b = b
        
    cpdef double evaluate(self, double x):
        return self.a * exp(self.b * x)


cdef class ActivationEnergyFunction(JumprateFunction):
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


cdef class LMCRoutine:
    """Main component of the cMD/LMC algorithm.
    This class determines the jump rates within a molecule according to the atomic distances and
    carries out the Monte Carlo steps."""

    cdef:
        double [:, :, ::1] oxygen_trajectory
        double [:, :, ::1] phosphorus_trajectory
        public int[:] phosphorus_neighbors
        gsl_rng * r
        public int jumps
        # Create containers for the oxygen index from which the proton jump start, for the index of
        # the destination oxygen, and for the jump probability of the oxygen connection.
        # The nested vectors hold the indices and probabilities for the whole trajectory
        public vector[vector[np.int32_t]] start_indices
        public vector[vector[np.int32_t]] destination_indices
        public vector[vector[np.float32_t]] jump_probability
        public vector[vector[int]] neighbors
        int oxygen_number, phosphorus_number
        double cutoff_radius, angle_threshold
        double neighbor_search_radius
        public JumprateFunction jumprate_fct
        public AtomBox atom_box
        np.uint32_t saved_frame_counter
        public double[:, ::1] jumpmatrix
        bool calculate_jumpmatrix

    def __cinit__(self, double [:, :, ::1] oxygen_trajectory, double [:, :, ::1] phosphorus_trajectory,
                  AtomBox atom_box, jumprate_parameter_dict, double cutoff_radius,
                  double angle_threshold, double neighbor_search_radius, jumprate_type, *,
                  seed=None, bool calculate_jumpmatrix=False, verbose=False):
        cdef:
            int i
            double[:] pbc_extended
        self.r = gsl_rng_alloc(gsl_rng_mt19937)
        self.jumps = 0
        self.cutoff_radius = cutoff_radius
        self.angle_threshold = angle_threshold
        self.neighbor_search_radius = neighbor_search_radius
        self.saved_frame_counter = 0
        self.calculate_jumpmatrix = calculate_jumpmatrix
        self.jumpmatrix = np.zeros((self.oxygen_number, self.oxygen_number))
        self.atom_box = atom_box
        self.oxygen_trajectory = oxygen_trajectory
        self.phosphorus_trajectory = phosphorus_trajectory
        self.oxygen_number = self.oxygen_trajectory.shape[1] * self.atom_box.box_multiplier[0] * \
                             self.atom_box.box_multiplier[1] * self.atom_box.box_multiplier[2]
        self.phosphorus_number = self.phosphorus_trajectory.shape[1] * self.atom_box.box_multiplier[0] * \
                                 self.atom_box.box_multiplier[1] * self.atom_box.box_multiplier[2]
        if type(seed) != int:
            seed = time.time()
        gsl_rng_set(self.r, seed)
        if verbose:
            print("# Using seed", seed)
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
            self.jumprate_fct = ActivationEnergyFunction(A, a, b, d0, T)
        elif jumprate_type == "Exponential_rates":
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            self.jumprate_fct = ExponentialFunction(a, b)
        else:
            raise Exception("Jump rate type unknown. Please choose between "
                            "MD_rates, Exponential_rates and AE_rates")
                            
    def __init__(self, double [:, :, ::1] oxygen_trajectory, double [:, :, ::1] phosphorus_trajectory,
                  AtomBox atom_box, jumprate_parameter_dict, double cutoff_radius,
                  double angle_threshold, double neighbor_search_radius, jumprate_type, *,
                  seed=None, bool calculate_jumpmatrix=False, verbose=False):

        self.phosphorus_neighbors = self.atom_box.determine_phosphorus_oxygen_pairs(self.oxygen_trajectory[0], self.phosphorus_trajectory[0])
        self.determine_neighbors(0)

    def __dealloc__(self):
        gsl_rng_free(self.r)

    def determine_neighbors(self, int frame_number, verbose=False):
        cdef:
            int i, j
            int oxygen_len = self.oxygen_trajectory.shape[1]
            double dist
            vector[int] neighbor_list

        self.neighbors.clear()
        for i in range(self.oxygen_number):
            neighbor_list.clear()
            for j in range(self.oxygen_number):
                if i != j:
                    dist = self.atom_box.length_extended_box_ptr(i, &self.oxygen_trajectory[
                        frame_number, 0, 0], oxygen_len,
                        j, &self.oxygen_trajectory[frame_number, 0, 0], oxygen_len)
                    if dist < self.neighbor_search_radius:
                        neighbor_list.push_back(j)
            self.neighbors.push_back(neighbor_list)

    cdef calculate_transitions(self, int frame_number, double r_cut, double angle_thresh):
        cdef:
            int start_index, neighbor_index, destination_index
            int oxygen_number_unextended = self.oxygen_trajectory.shape[1]
            int phosphorus_number_unextended = self.phosphorus_trajectory.shape[1]
            double dist, PO_angle
            vector[np.int32_t] start_indices_tmp
            vector[np.int32_t] destination_indices_tmp
            vector[np.float32_t] jump_probability_tmp

        start_indices_tmp.clear()
        destination_indices_tmp.clear()
        jump_probability_tmp.clear()
        for start_index in range(self.oxygen_number):
            for neighbor_index in range(self.neighbors[start_index].size()):
                destination_index = self.neighbors[start_index][neighbor_index]
                dist = self.atom_box.length_extended_box_ptr(
                    start_index, &self.oxygen_trajectory[frame_number, 0, 0],
                    oxygen_number_unextended, destination_index,
                    &self.oxygen_trajectory[frame_number, 0, 0], oxygen_number_unextended)
                if dist < r_cut:
                    poo_angle = self.atom_box.angle_extended_box(
                        self.phosphorus_neighbors[start_index],
                        &self.phosphorus_trajectory[frame_number, 0, 0],
                        phosphorus_number_unextended,
                        start_index,
                        &self.oxygen_trajectory[frame_number, 0, 0],
                        oxygen_number_unextended,
                        destination_index,
                        &self.oxygen_trajectory[frame_number, 0, 0],
                        oxygen_number_unextended)
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

    def get_transition_number(self):
        return self.start_tmp.size()

    def jumprate_test(self, double x):
        return self.jumprate_fct.evaluate(x)

    def print_transitions(self, int frame):
        cdef int i
        for i in range(self.start_indices[frame].size()):
            print(self.start_indices[frame][i], self.destination_indices[frame][i],
                   self.jump_probability[frame][i])
        print("In total", self.start_indices[frame].size(), "connections")

    def return_transitions(self, int frame):
        return self.start_indices[frame], self.destination_indices[frame], self.jump_probability[frame]

    def store_jumprates(self, bool verbose=False):
        cdef int i
        for i in range(self.oxygen_trajectory.shape[0]):
            self.calculate_transitions(i, self.cutoff_radius, self.angle_threshold)
            if verbose and i % 1000 == 0:
                print ("# Saving transitions {} / {}".format(i, self.oxygen_trajectory.shape[0]),
                       end="\r")
        print("")
        if verbose:
            print("# Done")

    def sweep(self, int frame, np.uint8_t[:] proton_lattice):
        cdef:
            int step, i, index_origin, index_destination
            int trajectory_length
            int steps

        steps = self.start_indices[frame].size()

        for step in range(steps):
            i = gsl_rng_uniform_int(self.r, steps)
            index_origin = self.start_indices[frame][i]
            index_destination = self.destination_indices[frame][i]
            if proton_lattice[index_origin] > 0 and proton_lattice[index_destination] == 0:
                if gsl_rng_uniform(self.r) < self.jump_probability[frame][i]:
                    proton_lattice[index_destination] = proton_lattice[index_origin]
                    proton_lattice[index_origin] = 0
                    self.jumps += 1

    def sweep_with_jumpmatrix(self, int frame, np.uint8_t[:] proton_lattice):
        cdef:
            int step, i, index_origin, index_destination
            int trajectory_length
            int steps
        trajectory_length = self.atom_box.oxygen_trajectory.shape[0]
        while self.saved_frame_counter < trajectory_length and self.saved_frame_counter < frame + 1:
            self.calculate_transitions(
                self.saved_frame_counter, self.cutoff_radius, self.angle_threshold)
        steps = self.start_indices[frame].size()
        for step in range(steps):
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
