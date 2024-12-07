#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=main
#SBATCH --array=0-99          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=../slurm/%A-%a.out

module load gurobi/10.0.0
source /home1/caijunya/.bashrc
conda activate backdoor_cl
# Run the Python script with the appropriate instance file
# python evaluate.py --instance ../instance/CA_2000_4000/valid/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA" --models 200 multitask_200 50 multitask_50
# python evaluate.py --instance ../instance/CA_3000_6000/valid/CA_3000_6000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_h" --models 200 multitask_200

# python evaluate.py --instance ../instance/INDSET_BA4_3000/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "IS" --models multitask_50
python evaluate.py --instance ../instance/INDSET_BA5_6000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_hh" --models 200 multitask_200 50 multitask_50

# python evaluate.py --instance ../instance/MVC_BA5_3000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC" --models multitask_50
# python evaluate.py --instance ../instance/MVC_BA5_6000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_h" --models multitask_50