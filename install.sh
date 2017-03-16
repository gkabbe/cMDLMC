#!/bin/bash

# This install helper tries to install the listed packages in requirements.txt using
# the conda package manager first, and pip whenever the conda installation fails

while read package
do
    conda install --yes $package || pip install $package
done < requirements.txt
