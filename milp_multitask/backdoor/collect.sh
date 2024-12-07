#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=main
#SBATCH --array=1-200          # Create a job array from 1 to 200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=../slurm/%A-%a.out

module load gurobi
source /home1/caijunya/.bashrc
conda activate backdoor_cl
# python backdoor_search/backdoor_search.py --instance_dir ../instance/CA_175_850/train/ca_${SLURM_ARRAY_TASK_ID}.lp --max_time 18000 --parallel 10 --exp_name "CA_175_850"
# python backdoor_search/backdoor_evaluate.py --instance_dir ../instance/CA_175_850/train/ca_${SLURM_ARRAY_TASK_ID}.lp --exp_name "CA_175_850"
# python backdoor_search/backdoor_search.py --instance_dir ../instance/INDSET_ER_1250/train/is_${SLURM_ARRAY_TASK_ID}.lp --max_time 18000 --parallel 10 --exp_name "INDSET_ER_1250"
# python backdoor_search/backdoor_evaluate.py --instance_dir ../instance/INDSET_ER_1250/train/is_${SLURM_ARRAY_TASK_ID}.lp --exp_name "INDSET_ER_1250"
python backdoor_search/backdoor_search.py --instance_dir ../instance/MVC_BA5_1500/train/mvc_${SLURM_ARRAY_TASK_ID}.lp --max_time 18000 --parallel 10 --exp_name "MVC_BA5_1500"
python backdoor_search/backdoor_evaluate.py --instance_dir ../instance/MVC_BA5_1500/train/mvc_${SLURM_ARRAY_TASK_ID}.lp --exp_name "MVC_BA5_1500"