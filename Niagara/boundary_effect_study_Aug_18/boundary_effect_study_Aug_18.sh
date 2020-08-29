#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80 #could be 40
#SBATCH --time=1:00:00
#SBATCH	--job-name=boundary

cd $SLURM_SUBMIT_DIR
module restore confine_module #python, gnu parrallel
source ~/.virtualenvs/confine_envs/bin/activate #confinepy

timeout 58m parallel --joblog "$SLURM_JOBID.log" -j $SLURM_TASKS_PER_NODE "python main_script.py {1} {2}" ::: {10::28} ::: {30,60}
EXIT=$?

if [ $EXIT -eq 124 ]
then
    ssh -t nia-login01 "cd $SLURM_SUBMIT_DIR; sbatch boundary_effect_study_Aug_18.sh"
fi