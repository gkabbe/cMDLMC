MD/KMC algorithm
================

This Python package includes the MD/KMC algorithm described in
http://pubs.acs.org/doi/abs/10.1021/ct500482k

Requirements
------------

The MD/KMC package needs the `GNU Scientific
Library <http://www.gnu.org/software/gsl/>`__ for the random number
generation.

How to install
--------------

The recommended way to install the mdkmc package, is by creating a
`virtualenv <https://virtualenv.pypa.io/en/latest>`__ environment:

::

    virtualenv ~/virtualenvs/mdkmc_env

This creates an isolated Python environment for the MD/KMC algorithm.
Packages installed there need no root permission, nor do they conflict
with previous Python installations.

Afterwards, type

::

    source ~/virtualenvs/mdkmc_env/bin/activate

This sets all Python variables to the path in your virtual environment.

Next, install some packages, the MD/KMC algorithm needs:

::

    pip install cython numpy CythonGSL

Now you can install the package via

::

    python setup.py install

Usage
-----

Use ``mdmc`` in order to start the main program. A config file must be
specified. It looks like this:

::

    # Prototype of a kMC config file
    # Comments via '#' at the beginning of the line
    filename 400K.xyz
    mode ASEP
    sweeps 75000 
    equilibration_sweeps 2500000
    md_timestep_fs 0.4   # The timestep of the trajectory on which the KMC will run
    skip_frames 0
    print_freq 75
    reset_freq 25000
    dump_trajectory False
    box_multiplier 1 1 1
    neighbor_freq 10
    jumprate_params_fs dict(a=0.06, b=2.363, c=0.035)
    jumprate_type MD_rates
    proton_number 96
    pbc 29.122 25.354 12.363
    cutoff_radius 4.
    po_angle True
    verbose True

Another script included in this package is ``jumpstat``. It analyses the
jump probability between two oxygen atoms depending on their mutual
distance.
