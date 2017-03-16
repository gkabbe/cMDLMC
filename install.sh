#!/bin/bash

# This install helper tries to install the listed packages in requirements.txt using
# 

while read package
do
    conda install --yes $package || pip install $package
done < requirements.txt
