#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=main
#SBATCH --array=1-100      # Create a job array from 1 to 100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=../slurm/%A-%a.out

module load gurobi/10.0.0
source /home1/caijunya/.bashrc
conda activate backdoor_cl
# Run the Python script with the appropriate instance file
# python evaluate.py --instance ../instance/CA_2000_4000/valid/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA" --model pretrain/CA/model_best_epoch5866.pth --multitask_model pretrain/CA/model_best.pth 
# python evaluate.py --instance ../instance/CA_3000_6000/valid/CA_3000_6000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_h" --model pretrain/CA/model_best_epoch5866.pth --multitask_model pretrain/CA/model_best.pth 
# python evaluate.py --instance ../instance/CA_2000_4000/valid/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --expname "CAv2_s" --model pretrain/CA/model_best_epoch5866.pth --multitask_model pretrain/CA/model_best.pth --multitask_model_v2 pretrain/CA_multi_v2/model_best.pth 
# python evaluate.py --instance ../instance/CA_3000_6000/valid/CA_3000_6000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_hv2_s" --model pretrain/CA/model_best_epoch5866.pth --multitask_model pretrain/CA/model_best.pth --multitask_model_v2 pretrain/CA_multi_v2/model_best.pth 
# python evaluate.py --instance ../instance/CA_2000_4000/valid/CA_2000_4000_${SLURM_ARRAY_TASK_ID}.lp --expname "CAv3" --model pretrain/CA_50/model_best.pth --multitask_model pretrain/CA_multi_v2_50/model_best.pth 
# python evaluate.py --instance ../instance/CA_3000_6000/valid/CA_3000_6000_${SLURM_ARRAY_TASK_ID}.lp --expname "CA_hv3" --model pretrain/CA_50/model_best.pth --multitask_model pretrain/CA_multi_v2_50/model_best.pth 

# python evaluate.py --instance ../instance/INDSET_BA5_6000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "IS" --model pretrain/IS/model_best_epoch4314.pth --multitask_model pretrain/IS/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA5_9000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_h" --model pretrain/IS/model_best_epoch4314.pth --multitask_model pretrain/IS/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA5_6000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "ISv2" --model pretrain/IS/model_best_epoch4314.pth --multitask_model pretrain/IS/model_best.pth --multitask_model_v2 pretrain/IS_multi_v2/model_best.pth
# python evaluate.py --instance ../instance/INDSET_BA5_9000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_hv2" --model pretrain/IS/model_best_epoch4314.pth --multitask_model pretrain/IS/model_best.pth --multitask_model_v2 pretrain/IS_multi_v2/model_best.pth
python evaluate.py --instance ../instance/INDSET_BA5_6000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "ISv3" --model pretrain/IS_50/model_best.pth --multitask_model pretrain/IS_multi_v2_50/model_best.pth
python evaluate.py --instance ../instance/INDSET_BA5_9000/valid/IS_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "IS_hv3" --model pretrain/IS_50/model_best.pth --multitask_model pretrain/IS_multi_v2_50/model_best.pth

# python evaluate.py --instance ../instance/MVC_BA5_6000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "MVCv2" --model pretrain/MVC/model_best.pth --multitask_model pretrain/MVC_multi/model_best.pth --multitask_model_v2 pretrain/MVC_multi_v2/model_best.pth
# python evaluate.py --instance ../instance/MVC_BA5_9000/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_hv2" --model pretrain/MVC/model_best.pth --multitask_model pretrain/MVC_multi/model_best.pth --multitask_model_v2 pretrain/MVC_multi_v2/model_best.pth
# python evaluate.py --instance ../instance/MVC_BA5_6000/valid/MVC_barabasi_albert_${SLURM_ARRAY_TASK_ID}.lp --expname "MVCv3" --model pretrain/MVC_50/model_best.pth --multitask_model pretrain/MVC_multi_v2_50/model_best.pth 
# python evaluate.py --instance ../instance/MVC_BA5_9000/valid/mvc_${SLURM_ARRAY_TASK_ID}.lp --expname "MVC_hv3" --model pretrain/MVC_50/model_best.pth --multitask_model pretrain/MVC_multi_v2_50/model_best.pth 