import os
import sys
import subprocess
import numpy  # to get includes
import cython_gsl
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


def get_commit_hash():
    command = "git log -n 1 --format=%H%n%s%n%ad"
    try:
        commit_hash, commit_message, commit_date = subprocess.check_output(command.split()).strip().split("\n")
    except subprocess.CalledProcessError:
        print >> sys.stderr, "Command '{}' could not be executed successfully.".format(command)
    return commit_hash, commit_message, commit_date


def readme():
    with open('README.rst', 'r') as f:
        return f.read()


def find_packages():
    """Find all packages (i.e. folders with an __init__.py file inside)"""
    packages = []
    for root, _, files in os.walk("mdkmc"):
        for f in files:
            if f == "__init__.py":
                packages.append(root)
    return packages


def find_extensions():
    """Find all .pyx files in all directories"""
    extensions = []
    for root, _, files in os.walk("mdkmc"):
        for f in files:
            if f.endswith(".pyx"):
                extensions.append(os.path.join(root, f))
    return extensions

subpackages = find_packages()
cython_exts = find_extensions()
ext_modules = []

for ext_path in cython_exts:
    ext_name = ext_path.replace(os.path.sep, ".")[:-4]
    print ext_path, ext_name
    ext_modules.append(Extension(
        ext_name,
        [ext_path],
        libraries=["m"]+cython_gsl.get_libraries(),
        extra_compile_args=["-O3", "-Wall"],
        language="c++",
        library_dirs=[cython_gsl.get_library_dir()],
        include_dirs=[".", numpy.get_include(), cython_gsl.get_cython_include_dir()]
    ))

setup(name='mdkmc',
      version='0.1',
      description='Implementation of the MD/KMC algorithm in Python',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Molecular Dynamics :: Kinetic Monte Carlo',
      ],
      keywords='MD KMC Chemistry',
      url='http://github.com/',
      author='Gabriel Kabbe',
      author_email='gabriel.kabbe@chemie.uni-halle.de',
      license='GPLv3',
      packages=subpackages,
      install_requires=[
          'numpy',
          'ipdb',
          'cythongsl==0.2.1',
          'gitpython',
          'cython'
      ],
      dependency_links=['https://github.com/twiecki/CythonGSL/tarball/master#egg=cython_gsl-0.2.1'],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts': ['mdmc=mdkmc.kMC.MDMC:main',
                              'jumpstat=mdkmc.analysis.jumpstat:main',
                              'poo-occupation=mdkmc.analysis.phosphonic_group_occupation:main',
                              'md_jumpmatrix=mdkmc.analysis.jumpmat_anglecrit:main',
                              'print_version=mdkmc.print_version:print_version',
                              'avg_mdmc=mdkmc.kMC.average_MC_out:main',
                              'msd=mdkmc.analysis.msd:main',
                              'ohbond-corr=mdkmc.analysis.ohbond_autocorr:main'
                             ],
      },
      # scripts=["bin/MDMC.py"],
      include_package_data=True,
      zip_safe=False,
      # ext_modules=ext_modules,
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext}
      )

# Write hash of current commit to file
with open("mdkmc/version_hash.py", "w") as f:
    commit_hash, commit_message, commit_date = get_commit_hash()
    print >> f, "commit_hash = \"{}\"".format(commit_hash)
    print >> f, "commit_message = \"{}\"".format(commit_message)
    print >> f, "commit_date = \"{}\"".format(commit_date)

