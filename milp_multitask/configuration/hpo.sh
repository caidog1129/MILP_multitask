#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=main
#SBATCH --array=0-199        # Create a job array from 1 to 200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=15:00:00
#SBATCH --output=../slurm/%A-%a.out

module load gurobi/10.0.0
source /home1/caijunya/.bashrc
conda activate backdoor_cl
# Run the Python script with the appropriate instance file
# python hpo.py --instance ../instance/CA_2000_4000/train/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp
# python hpo.py --instance ../instance/CA_2000_4000/valid/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --exp_name "CA_test"
# python hpo.py --instance ../instance/CA_3000_6000/valid/CA_3000_6000_${SLURM_ARRAY_TASK_ID}.lp --exp_name "CA_test_h"

# python hpo.py --instance ../instance/INDSET_BA4_3000/train/is_${SLURM_ARRAY_TASK_ID}.lp
# python hpo.py --instance ../instance/INDSET_BA4_3000/valid/is_${SLURM_ARRAY_TASK_ID}.lp --exp_name "INDSET_test"
# python hpo.py --instance ../instance/INDSET_BA4_4500/valid/is_${SLURM_ARRAY_TASK_ID}.lp --exp_name "INDSET_test_h"

# python hpo.py --instance ../instance/MVC_BA5_3000/train/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp
# python hpo.py --instance ../instance/MVC_BA5_3000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --exp_name "MVC_test"
# python hpo.py --instance ../instance/MVC_BA5_6000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --exp_name "MVC_test_h"

# python hpo_gurobi.py --instance ../instance/CA_2000_4000/train/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --exp_name "CA_gurobi"
# python hpo_gurobi.py --instance ../instance/INDSET_BA5_6000/train/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --exp_name "MIS_gurobi"
python hpo_gurobi.py --instance ../instance/MVC_BA5_6000/train/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --exp_name "MVC_gurobi"
