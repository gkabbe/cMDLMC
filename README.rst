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

Next, install some packages (just copy and paste the following line):

::

    pip install cython; pip install numpy; pip install CythonGSL; pip install gitpython

Alternatively, just run

::

    pip install -r requirements.txt

Now you can install the mdkmc package via

::

    python setup.py install

Usage
-----

Use ``mdmc`` in order to start the main program. A config file must be
specified. All supported keywords are found by typing
``mdmc config_help``

Another script included in this package is ``jumpstat``. It analyses the
jump probability between two oxygen atoms depending on their mutual
distance.
