MD/KMC algorithm
================

This Python package includes the cMD/LMC algorithm described in 

[http://pubs.acs.org/doi/abs/10.1021/ct500482k](http://pubs.acs.org/doi/abs/10.1021/ct500482k)

[http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b05822](http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b05822)

and

[http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b05821](http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b05821)

Requirements
------------
The cMD/LMC package needs the [GNU Scientific Library](http://www.gnu.org/software/gsl/)
for the random number generation.

On Debian, you can install the GSL package via

    sudo apt install libgsl-dev


How to install
--------------
The recommended way to install the mdlmc package, is by using [conda](https://www.continuum.io/downloads)

    pyvenv ~/virtualenvs/mdkmc_env

This creates an isolated Python environment for the cMD/LMC algorithm. 
Packages can be installed in the virtual environment without root 
permission and are independent from system-wide Python installations.

Afterwards, type

    source ~/virtualenvs/mdkmc_env/bin/activate

This sets all Python variables to the path in your virtual environment.

Next, install some packages (just copy and paste the following line):

    pip install cython; pip install numpy; pip install CythonGSL; pip install gitpython

Now you can install the mdlmc package via

    python setup.py install

Usage
-----
Use `mdmc` in order to start the main program. A config file must be specified.
All supported keywords are found by typing `mdmc config_help`

Another script included in this package is `jumpstat`. It analyses the protob jump probability
between two oxygen atoms depending on their mutual distance.
