#!/bin/bash

pip install git+https://github.com/pdjely/pymdptoolbox.git
pushd .
cd src/A4/gym-tictactoe
pip install -e .
