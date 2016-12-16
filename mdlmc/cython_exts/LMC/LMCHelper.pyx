# cython: profile = False
# cython: boundscheck = False, wraparound = False, cdivision = True, initializedcheck = False
# cython: language_level = 3
import time

import numpy as np
import cython
import tables
import h5py

cimport numpy as np
cimport cython
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
    cpdef double evaluate(self, double distance):
        return 0


cdef class FermiFunction(JumprateFunction):
    """Calculates the jump probability according to the function
        f(x) = a / ( (1 + exp(x - b) / c)"""
    cdef:
        double a, b, c

    def __cinit__(self, double a, double b, double c):
        self.a = a
        self.b = b
        self.c = c

    cpdef double evaluate(self, double distance):
        return self.a / (1 + exp((distance - self.b) / self.c))


cdef class ExponentialFunction(JumprateFunction):
    """Calculates the jump probability according to the function
        f(x) = a * exp(b*x)"""
    cdef:
        double a, b
        
    def __cinit__(self, double a, double b):
        self.a = a
        self.b = b
        
    cpdef double evaluate(self, double distance):
        return self.a * exp(self.b * distance)


cdef class ActivationEnergyFunction(JumprateFunction):
    """Calculates the activation energy according to the function
        E(x) = a * (x - d0) / sqrt(b + 1 / (x -d0)**2)
    Finally, E(x) is inserted into the Arrhenius equation
        f(x, T) = A * exp(-E(x)/(k_B * T)"""
    cdef:
        double A, a, b, d0, T

    def __cinit__(self, double A, double a, double b, double d0, double T):
        self.A = A
        self.a = a
        self.b = b
        self.d0 = d0
        self.T = T

    cpdef double evaluate(self, double distance):
        cdef double E
        if distance <= self.d0:
            E = 0
        else:
            E = self.a * (distance - self.d0) / sqrt(self.b + 1.0 / (distance - self.d0) / (distance - self.d0))
        return self.A * exp(-E / (R * self.T))

    cpdef double get_energy(self, double distance):
        if distance <= self.d0:
            return 0
        else:
            return self.a * (distance - self.d0) / sqrt(self.b + 1.0 / (distance - self.d0) / (distance - self.d0))


cdef class AngleFunction:
    cpdef double evaluate(self, double angle_rad):
        return 0


cdef class AngleDummy(AngleFunction):
    cpdef double evaluate(self, double angle_rad):
        return 1


cdef class AngleCutoff(AngleFunction):
    # Angle threshold in radians
    cdef double theta_0

    def __cinit__(self, double theta_0):
        self.theta_0 = theta_0

    cpdef double evaluate(self, double angle_rad):
        return <double> angle_rad >= self.theta_0


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
        double cutoff_radius
        double neighbor_search_radius
        public JumprateFunction jumprate_fct
        public AngleFunction angle_fct
        public AtomBox atom_box
        np.uint32_t saved_frame_counter
        public double[:, ::1] jumpmatrix
        bool calculate_jumpmatrix

    def __cinit__(self, double [:, :, ::1] oxygen_trajectory, double [:, :, ::1] phosphorus_trajectory,
                  AtomBox atom_box, jumprate_parameter_dict, double cutoff_radius,
                  double angle_threshold, double neighbor_search_radius, jumprate_type, *,
                  neighbor_list=True, seed=None, bool calculate_jumpmatrix=False, verbose=False,
                  angle_dependency=True):
        cdef:
            int i
            double[:] pbc_extended
        self.r = gsl_rng_alloc(gsl_rng_mt19937)
        self.jumps = 0
        self.cutoff_radius = cutoff_radius
        self.neighbor_search_radius = neighbor_search_radius
        self.saved_frame_counter = 0
        self.atom_box = atom_box
        self.oxygen_trajectory = oxygen_trajectory
        self.phosphorus_trajectory = phosphorus_trajectory
        self.oxygen_number = self.oxygen_trajectory.shape[1] * self.atom_box.box_multiplier[0] * \
                             self.atom_box.box_multiplier[1] * self.atom_box.box_multiplier[2]
        self.phosphorus_number = self.phosphorus_trajectory.shape[1] * self.atom_box.box_multiplier[0] * \
                                 self.atom_box.box_multiplier[1] * self.atom_box.box_multiplier[2]
        self.calculate_jumpmatrix = calculate_jumpmatrix
        self.jumpmatrix = np.zeros((self.oxygen_number, self.oxygen_number))

        if type(seed) != int:
            seed = time.time()
        gsl_rng_set(self.r, seed)
        if verbose:
            print("# Using seed", seed)

        # Jump rates determined via jumpstat
        if jumprate_type == "MD_rates":
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            c = jumprate_parameter_dict["c"]
            self.jumprate_fct = FermiFunction(a, b, c)

        # Jump rates determined via energy surface scans
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

        if angle_dependency:
            self.angle_fct = AngleCutoff(angle_threshold)
        else:
            self.angle_fct = AngleDummy()

    def __init__(self, double [:, :, ::1] oxygen_trajectory, double [:, :, ::1] phosphorus_trajectory,
                 AtomBox atom_box, jumprate_parameter_dict, double cutoff_radius,
                 double angle_threshold, double neighbor_search_radius, jumprate_type, *,
                 neighbor_list=True, seed=None, bool calculate_jumpmatrix=False, verbose=False,
                 angle_dependency=True):

        self.phosphorus_neighbors = self.atom_box.determine_phosphorus_oxygen_pairs(
            self.oxygen_trajectory[0], self.phosphorus_trajectory[0])
        if neighbor_list:
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

    cdef calculate_transitions_without_neighborlist(self, int frame_number, double r_cut):
        cdef:
            int start_index, destination_index
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
            for destination_index in range(start_index):

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
                    jump_probability_tmp.push_back(
                        self.jumprate_fct.evaluate(dist) * self.angle_fct.evaluate(poo_angle))

        self.start_indices.push_back(start_indices_tmp)
        self.destination_indices.push_back(destination_indices_tmp)
        self.jump_probability.push_back(jump_probability_tmp)
        self.saved_frame_counter += 1

    cdef calculate_transitions_with_neighborlist(self, int frame_number, double r_cut):
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
                    jump_probability_tmp.push_back(
                        self.jumprate_fct.evaluate(dist) * self.angle_fct.evaluate(poo_angle))

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

    def store_jumprates(self, *, bool use_neighborlist=False, bool verbose=False):
        cdef int i
        for i in range(self.oxygen_trajectory.shape[0]):
            if use_neighborlist:
                self.calculate_transitions_with_neighborlist(i, self.cutoff_radius)
            else:
                self.calculate_transitions_without_neighborlist(i, self.cutoff_radius)

            if verbose and i % 1000 == 0:
                print ("# Saving transitions {} / {}".format(i, self.oxygen_trajectory.shape[0]),
                       end="\r", flush=True)
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


@cython.final
cdef class KMCRoutine:

    cdef:
        JumprateFunction jumprate_fct
        gsl_rng * r
        double cutoff_radius
        AtomBox atombox
        public object oxygen_trajectory_hdf5

        vector[vector[double]] jump_probability_per_frame
        double abc

    def __cinit__(self, oxygen_trajectory_hdf5: "Trajectory in HDF5 format", AtomBox atombox,
                  JumprateFunction jumprate_fct):
        self.atombox = atombox
        self.jumprate_fct = jumprate_fct

    def __init__(self, oxygen_trajectory_hdf5: "Trajectory in HDF5 format", AtomBox atombox,
                 JumprateFunction jumprate_fct):
        self.oxygen_trajectory_hdf5 = oxygen_trajectory_hdf5

    def determine_probability_sums(self, double [:, :, ::1] oxygen_trajectory):
        cdef:
            int f, i, j
            np.ndarray[np.float32_t, ndim=2] probs

        probs = np.zeros((oxygen_trajectory.shape[0], oxygen_trajectory.shape[1]), dtype=np.float32)

        for f in range(oxygen_trajectory.shape[0]):
            for i in range(oxygen_trajectory.shape[1]):
                for j in range(oxygen_trajectory.shape[1]):
                    dist = self.atombox.length_(oxygen_trajectory[f, j], oxygen_trajectory[f, i])
                    if i != j:
                        probs[f, i] += self.jumprate_fct.evaluate(dist)
        return probs
