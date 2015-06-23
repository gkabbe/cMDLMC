from mdkmc.cython_exts.helper cimport  math_helper
from mdkmc.cython_exts.kMC.kMC_helper cimport Function, AEFunction, FermiFunction
from mdkmc.cython_exts.atoms cimport numpyatom as cnpa
cimport numpy as np
from libcpp.vector cimport vector

cdef class JumprateStorage:
    cdef:
        int saved_framenumbers
        int trajectory_length
        double [:, :, ::1] trajectory
        double [:] pbc
        double r_cut
        Function jumprate_function
        vector[int] start_tmp
        vector[int] destination_tmp
        vector[double] jump_probability_tmp
        vector[vector[int]] start
        vector[vector[int]] destination
        vector[vector[double]] jump_probability


    def __cinit__(self, double [:, :, ::1] trajectory, double [:] pbc, jumprate_parameter_dict):
        self.trajectory = trajectory
        self.pbc = pbc
        self.trajectory_length = trajectory.shape[0]
        self.saved_framenumbers = 0
        self.r_cut = 4.0

        if "a" in jumprate_parameter_dict and "b" in jumprate_parameter_dict and "c" in jumprate_parameter_dict:
            a = jumprate_parameter_dict["a"]
            b = jumprate_parameter_dict["b"]
            c = jumprate_parameter_dict["c"]
            self.jumprate_function = FermiFunction(a, b, c)
        elif "A" in jumprate_parameter_dict and "a" in jumprate_parameter_dict \
            and "x0" in jumprate_parameter_dict and "xint" in jumprate_parameter_dict \
            and "T" in jumprate_parameter_dict:

            A = jumprate_parameter_dict["A"]
            a = jumprate_parameter_dict["a"]
            x0 = jumprate_parameter_dict["x0"]
            xint = jumprate_parameter_dict["xint"]
        else:
            raise ValueError("Wrong parameters in jumprate_parameter_dict")

    cdef save_next_frame_jumprates(self):
        cdef:
            int i, j
            double [:, ::1] frame = self.trajectory[self.saved_framenumbers]
            double dist

        self.start_tmp.clear()
        self.destination_tmp.clear()
        self.jump_probability_tmp.clear()

        for i in range(frame.shape[0]):
            for j in range(frame.shape[0]):
                if i != j:
                    dist = cnpa.length_ptr(&frame[i, 0], &frame[j, 0], &self.pbc[0])
                    if dist < self.r_cut:
                        self.start_tmp.push_back(i)
                        self.destination_tmp.push_back(j)
                        self.jump_probability_tmp.push_back(self.jumprate_function.evaluate(dist))
        self.start.push_back(self.start_tmp)
        self.destination.push_back(self.destination_tmp)
        self.jump_probability.push_back(self.jump_probability_tmp)
        self.saved_framenumbers += 1

    def get_values(self, int framenumber):
        cdef int i

        if framenumber > self.trajectory_length:
            raise IndexError("Requested index larger than trajectory length")
        else:
            while self.saved_framenumbers < framenumber + 1:
                self.save_next_frame_jumprates()

            for i in range(self.start.size()):
                print self.start[framenumber][i], self.destination[framenumber][i], self.jump_probability[framenumber][i]

    def int_test(self):
        cdef np.int8_t i = 3
        print i


