#!/usr/bin/python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy # to get includes
import cython_gsl

import os
import sys

sys.path.append("/usr/lib/python2.7/dist-packages/Cython/Includes/libcpp")
sys.path.append("/home/kabbe/PhD/pythontools/cython_exts/atoms")

# exts = ["jumpstat_helper", "kMC_helper"]
exts = [f[:-4] for f in os.listdir(".") if ".pyx" in f]

for ext_name in exts:
    setup(
            cmdclass = {'build_ext': build_ext},
            ext_modules = [Extension(ext_name, [ext_name+".pyx"],
                                     libraries=["m"]+cython_gsl.get_libraries(),
                                     library_dirs=[cython_gsl.get_library_dir()],
                                     extra_compile_args=['-fopenmp'],
                                     extra_link_args=['-fopenmp'],
                                     language="c++"
                                     )
                           ],
            include_dirs = [numpy.get_include(),cython_gsl.get_cython_include_dir()],
            )
