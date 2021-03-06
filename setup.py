# coding=utf-8

import os
import sys
import subprocess
import numpy  # to get includes
import cython_gsl
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def get_commit_hash():
    command = "git log -n 1 --format=%H%n%s%n%ad"
    try:
        commit_hash, commit_message, commit_date = subprocess.check_output(
            command.split()).strip().split(b"\n")
    except subprocess.CalledProcessError:
        print("Command '{}' could not be executed successfully.".format(command), file=sys.stderr)
    return commit_hash.decode(), commit_message.decode(), commit_date.decode()


def readme():
    with open('README.rst', 'r') as f:
        return f.read()


def find_packages():
    """Find all packages (i.e. folders with an __init__.py file inside)"""
    packages = []
    for root, _, files in os.walk("mdlmc"):
        for f in files:
            if f == "__init__.py":
                packages.append(root)
    return packages


def find_extensions():
    """Find all .pyx files in all directories"""
    extensions = []
    for root, _, files in os.walk("mdlmc"):
        for f in files:
            if f.endswith(".pyx"):
                extensions.append(os.path.join(root, f))
    return extensions


subpackages = find_packages()
cython_exts = find_extensions()
ext_modules = []

for ext_path in cython_exts:
    ext_name = ext_path.replace(os.path.sep, ".")[:-4]
    print(ext_path, ext_name)
    ext_modules.append(Extension(
        ext_name,
        [ext_path],
        libraries=["m"] + cython_gsl.get_libraries(),
        extra_compile_args=["-O3", "-Wall", "-ffast-math"],
        language="c++",
        library_dirs=[],
        include_dirs=[".", numpy.get_include()]
    ))

setup(name='mdlmc',
      version='0.1',
      description='Implementation of the cMD/LMC algorithm in Python',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
      keywords='MD LMC KMC Chemistry',
      url='http://github.com/',
      author='Gabriel Kabbe',
      author_email='gabriel.kabbe@chemie.uni-halle.de',
      license='GPLv3',
      packages=subpackages,
      install_requires=['numpy', 'ipdb', 'cythongsl>=0.2.1', 'gitpython', 'cython', 'pint>=0.7.2',
                        'scipy', 'matplotlib'],
      python_requires='~=3.6',
      dependency_links=[],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts': ['mdmc=mdlmc.main:main',
                              'trajconv=mdlmc.IO.converters:main',
                              'mdlmc_config=mdlmc.config:main'
                              ],
      },
      include_package_data=True,
      zip_safe=False,
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext}
      )

# Write hash, message and date of current commit to file
with open("mdlmc/version_hash.py", "w") as f:
    commit_hash, commit_message, commit_date = get_commit_hash()
    print("commit_hash = \"{}\"".format(commit_hash), file=f)
    print("commit_message = \"{}\"".format(commit_message), file=f)
    print("commit_date = \"{}\"".format(commit_date), file=f)
