# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# cython: language_level = 3

import cython
import time

cimport numpy as np
import numpy as np

cimport mdlmc.cython_exts.atoms.numpyatom as cnpa

cdef double PI = np.pi

def get_anglematrix(double [:, :, ::1] Opos, double [:, :, ::1] Ppos, np.int64_t [::1] P_neighbors, double [::1] pbc,
                    np.int64_t [:, ::1] anglemat, double distance_cutoff=3.0, double angle_cutoff=PI/2):
    cdef:
        int i, O_i, O_j
        double distance, angle

    start_time = time.time()
    for i in xrange(Opos.shape[0]):
        for O_i in xrange(Opos.shape[1]):
            for O_j in xrange(Opos.shape[1]):
                if O_i != O_j:
                    distance = cnpa.length_ptr(&Opos[i, O_i, 0], &Opos[i, O_j, 0], &pbc[0])
                    if distance < 3.0:
                        angle = cnpa.angle(Opos[i, O_i], Ppos[i, P_neighbors[O_i]], Opos[i, O_i], Opos[i, O_j], pbc)
                        if angle >= angle_cutoff:
                            # print angle
                            # print "P", " ".join(map(str, Ppos[i, P_neighbors[O_i]]))
                            # print "O", " ".join(map(str, Opos[i, O_i]))
                            # print "O", " ".join(map(str, Opos[i, O_j]))
                            anglemat[O_i, O_j] = True
        if i % 1000 == 0:
            if i != 0:
                print("frame {:06d} -- {:.2f} sec. left".format(i, (time.time() - start_time)/i*(Opos.shape[0]-i)), end="\r")
    print("")