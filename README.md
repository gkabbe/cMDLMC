MD/KMC algorithm
================

This Python package includes the MD/KMC algorithm described in 
[http://pubs.acs.org/doi/abs/10.1021/ct500482k](http://pubs.acs.org/doi/abs/10.1021/ct500482k)


Requirements
------------
The MD/KMC package needs the [GNU Scientific Library](http://www.gnu.org/software/gsl/)
for the random number generation.


How to install
--------------
The recommended way to install the mdkmc package, is by creating a [virtualenv](https://virtualenv.pypa.io/en/latest) environment:

    virtualenv ~/virtualenvs/mdkmc_env

This creates an isolated Python environment for the MD/KMC algorithm. Packages installed there need no root permission, nor do they conflict
with previous Python installations.

Afterwards, type

    source ~/virtualenvs/mdkmc_env/bin/activate

This sets all Python variables to the path in your virtual environment.

Next, install some packages, the MD/KMC algorithm needs:

    pip install cython numpy CythonGSL

Now you can install the package via

    python setup.py install

Usage
-----
