#!/bin/bash 
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=2:00:00
#SBATCH --job-name=csN7
#SBATCH --output=csN7_output.txt
#SBATCH --mail-type=FAIL
 
cd $SLURM_SUBMIT_DIR
 
module load NiaEnv/2019b
module load python/3.6.8

python "run_N7p1.py" &
python "run_N7p2.py" &
python "run_N7p3.py" &