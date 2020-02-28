#!/bin/bash 
 
module load NiaEnv/2019b
module load python/3.6.8

python "run_N7p1.py" &
python "run_N7p2.py" &
python "run_N7p3.py" &