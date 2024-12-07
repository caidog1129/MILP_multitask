#!/bin/bash
#SBATCH --account=dilkina_438
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/%A-%a.out

module load gurobi/10.0.0
source /home1/caijunya/.bashrc
mamba activate backdoor_cl

# python train.py --taskName "CA_200" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/ 
# python train.py --taskName "CA_50" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/ --fracdata 0.25
# python train.py --taskName "CA_multitask_50" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/ --pretrainModel ../multitask/pretrain/CA/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "IS_200" --instance_dir ../instance/INDSET_BA4_3000/train/ --result_dir datasets/INDSET_BA4_3000/
# python train.py --taskName "IS_50" --instance_dir ../instance/INDSET_BA4_3000/train/ --result_dir datasets/INDSET_BA4_3000/ --fracdata 0.25
python train.py --taskName "IS_multitask_50" --instance_dir ../instance/INDSET_BA4_3000/train/ --result_dir datasets/INDSET_BA4_3000/ --pretrainModel ../multitask/pretrain/IS/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "MVC_200" --instance_dir ../instance/MVC_BA5_3000/train/ --result_dir datasets/MVC_BA5_3000/
# python train.py --taskName "MVC_50" --instance_dir ../instance/MVC_BA5_3000/train/ --result_dir datasets/MVC_BA5_3000/ --fracdata 0.25
# python train.py --taskName "MVC_multitask_50" --instance_dir ../instance/MVC_BA5_3000/train/ --result_dir datasets/MVC_BA5_3000/ --pretrainModel ../multitask/pretrain/MVC/model_best.pth --freeze 1 --fracdata 0.25