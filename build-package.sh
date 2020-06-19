#!/bin/bash
## Script to deploy PyPI package

# Build distribution files locally
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# Push up distribution files to PyPI
python3 -m pip install --user --upgrade twine
python3 -m twine upload dist/*
