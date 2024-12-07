#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=main
#SBATCH --array=1-100       # Create a job array from 1 to 100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=../slurm/%A-%a.out

module load gurobi/10.0.0
source /home1/caijunya/.bashrc
conda activate backdoor_cl
# Run the Python script with the appropriate instance file
# python evaluate.py --instance ../instance/CA_175_850/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CA" --model pretrain/CA_175_850/model_best_epoch1493.pth --multitask_model pretrain/CA_175_850/model_best.pth
# python evaluate.py --instance ../instance/CA_200_1000/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_h" --model pretrain/CA_175_850/model_best_epoch1493.pth --multitask_model pretrain/CA_175_850/model_best.pth
# python evaluate.py --instance ../instance/CA_175_850/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CAv3" --model pretrain/CA_175_850/model_best_epoch1493.pth --multitask_model pretrain/CA_175_850/model_best.pth --multitask_model_v3 pretrain/CA_175_850_multi_v3/model_best.pth
# python evaluate.py --instance ../instance/CA_200_1000/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_hv3" --model pretrain/CA_175_850/model_best_epoch1493.pth --multitask_model pretrain/CA_175_850/model_best.pth --multitask_model_v3 pretrain/CA_175_850_multi_v3/model_best.pth
# python evaluate.py --instance ../instance/CA_175_850/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CAv2" --model pretrain/CA_175_850_50/model_best.pth --multitask_model pretrain/CA_175_850_multi_v3_50/model_best.pth
# python evaluate.py --instance ../instance/CA_200_1000/valid/ca_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_hv2" --model pretrain/CA_175_850_50/model_best.pth --multitask_model pretrain/CA_175_850_multi_v3_50/model_best.pth

# python evaluate.py --instance ../instance/INDSET_BA4_1250/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "IS" --model pretrain/INDSET_BA4_1250/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA4_1500/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_h" --model pretrain/INDSET_BA4_1250/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA4_1250/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "ISv3" --model pretrain/INDSET_BA4_1250/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi/model_best.pth --multitask_model_v3 pretrain/INDSET_BA4_1250_multi_v3/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA4_1500/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_hv3" --model pretrain/INDSET_BA4_1250/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi/model_best.pth --multitask_model_v3 pretrain/INDSET_BA4_1250_multi_v3/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA4_1250/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "ISv2" --model pretrain/INDSET_BA4_1250_50/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi_v3_50/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA4_1500/valid/is_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_hv2" --model pretrain/INDSET_BA4_1250_50/model_best.pth --multitask_model pretrain/INDSET_BA4_1250_multi_v3_50/model_best.pth

# python evaluate.py --instance ../instance/MVC_BA5_1500/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC" --model pretrain/MVC_BA5_1500/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi/model_best.pth
# python evaluate.py --instance ../instance/MVC_BA5_2000/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_h" --model pretrain/MVC_BA5_1500/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi/model_best.pth
# python evaluate.py --instance ../instance/MVC_BA5_1500/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVCv3" --model pretrain/MVC_BA5_1500/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi/model_best.pth --multitask_model_v3 pretrain/MVC_BA5_1500_multi_v3/model_best.pth
# python evaluate.py --instance ../instance/MVC_BA5_2000/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_hv3" --model pretrain/MVC_BA5_1500/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi/model_best.pth --multitask_model_v3 pretrain/MVC_BA5_1500_multi_v3/model_best.pth
python evaluate.py --instance ../instance/MVC_BA5_1500/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVCv2" --model pretrain/MVC_BA5_1500_50/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi_v3_50/model_best.pth
python evaluate.py --instance ../instance/MVC_BA5_2000/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_hv2" --model pretrain/MVC_BA5_1500_50/model_best.pth --multitask_model pretrain/MVC_BA5_1500_multi_v3_50/model_best.pth