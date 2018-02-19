#!/bin/sh

sphinx-apidoc -f -o sphinx/source mdlmc
sphinx-build -a -b html sphinx/source docs
