# cs7641-ml

Requires
========
Conda

Install
=======
1. Clone repository: git clone https://github.com/pdjely/cs7641-ml.git
2. Install conda dependencies: conda env create -f environment.yaml


Activate Environment
====================
cd cs7641-ml
conda activate cs7641

Assignment 3
============
checkout a3
cd src
python assignment3.py


****************** PREVIOUS ASSIGNMENTS *********************************
Assignment 2
============

# Install MLRose pdjely fork (first run only)
pip install git+https://github.com/pdjely/mlrose.git

# Run all experiments
cd src
python assignment2.py

----------------
A Note on MLRose
----------------
This project uses my own fork of the MLRose repository. This fork is located at
https://github.com/pdjely/mlrose. For this fork, I took modifications from
fellow students David Park (fast MIMIC) and Andrew Rollings (modified GA). I
did not use any of Andrews' other refactors, including any of his runner
code.


Assignment 1
============

# Check out Assignment 1
git checkout a1

# Run full experiment
cd src
python assignment1.py -d musk shoppers
