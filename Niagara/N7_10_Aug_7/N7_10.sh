#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80 #could be 40
#SBATCH --time=1:00:00
#SBATCH	--job-name=N7_10
#SBATCH --mail-user=samuelsy.wong@mail.utoronto.ca
#SBATCH --mail-type=ALL #email you everything

cd $SLURM_SUBMIT_DIR
module restore confine_module #python, gnu parrallel
source ~/.virtualenvs/confine_envs/bin/activate #confinepy

timeout 59m parallel --joblog "$SLURM_JOBID.log" -j $SLURM_TASKS_PER_NODE "python main_script.py {1} {2} {3}" ::: {7,10} ::: {1..5} ::: $(seq 50 -5 5)
EXIT=$?

if [ $EXIT -eq 124 ]
then
    ssh -t nia-login01 "cd $SLURM_SUBMIT_DIR; sbatch N7_10.sh"
fi