#!/bin/bash 

module load NiaEnv/2019b
module load python/3.6.8

python "test_N3p1.py" &
python "test_N3p2.py" &
