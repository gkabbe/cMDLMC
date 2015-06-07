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

Afterwards, type

    source ~/virtualenvs/mdkmc_env/bin/activate

Now you can install the package via

    python setup.py install

This will install the packages into ~/virtualenvs/mdkmc_env.

At the moment, the first installation will fail with the error
'ImportError: No module named cython_gsl'.

Just run the installation again, and everything should work fine.


Usage
-----
