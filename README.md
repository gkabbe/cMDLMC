cMD/LMC algorithm
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

The recommended Python version is 3.5


How to install
--------------
The recommended way to install the mdlmc package, is by using a virtual environment.
This can be achieved using [conda](https://www.continuum.io/downloads) or pyvenv, which comes 
with the Python 3 interpreter.
This way, packages can be installed in the virtual environment without root 
permission and are independent from system-wide Python installations.

    pyvenv ~/virtualenvs/mdlmc_env

This creates an isolated Python environment for the cMD/LMC algorithm. 

Afterwards, type

    source ~/virtualenvs/mdlmc_env/bin/activate

This sets all Python variables to the path in your virtual environment.

Next, install some packages (just copy and paste the following line):

    pip install cython; pip install numpy; pip install CythonGSL; pip install gitpython

Now you can install the mdlmc package via

    python setup.py install

Usage
-----
Use `mdmc config_load <config file>` in order to start the main program. The path of a config file must be specified instead of <config file>.
All supported keywords are found by typing `mdmc config_help`

Another script included in this package is `jumpstat`. It analyses the proton jump probability
between two oxygen atoms depending on their mutual distance.
