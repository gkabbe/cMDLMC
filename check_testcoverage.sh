#!/bin/bash

coverage run --source mdlmc -m pytest tests
coverage html -i
chromium htmlcov/index.html
