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
# python train.py --taskName "CA_50" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/solution/ --fracdata 0.25
# python train.py --taskName "CA" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/solution/ --pretrainModel ../multitask/pretrain/CA/model_best.pth --freeze 1
# python train.py --taskName "CA_multi_v2_50" --instance_dir ../instance/CA_2000_4000/train/ --result_dir datasets/CA_2000_4000/solution/ --pretrainModel ../multitask/pretrain/CA_v2/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "IS_50" --instance_dir ../instance/INDSET_BA5_6000/train/ --result_dir datasets/INDSET_BA5_6000/solution/ --fracdata 0.25
# python train.py --taskName "IS" --instance_dir ../instance/INDSET_BA5_6000/train/ --result_dir datasets/INDSET_BA5_6000/solution/ --pretrainModel ../multitask/pretrain/IS/model_best.pth --freeze 1
# python train.py --taskName "IS_multi_v2_50" --instance_dir ../instance/INDSET_BA5_6000/train/ --result_dir datasets/INDSET_BA5_6000/solution/ --pretrainModel ../multitask/pretrain/IS_v2/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "MVC_50" --instance_dir ../instance/MVC_BA5_6000/train/ --result_dir datasets/MVC_BA5_6000/solution/ --fracdata 0.25
# python train.py --taskName "MVC_multi" --instance_dir ../instance/MVC_BA5_6000/train/ --result_dir datasets/MVC_BA5_6000/solution/ --pretrainModel ../multitask/pretrain/MVC/model_best.pth --freeze 1
# python train.py --taskName "MVC_multi_x3" --instance_dir ../instance/MVC_BA5_6000/train/ --result_dir datasets/MVC_BA5_6000/solution/ --pretrainModel ../multitask/pretrain/MVC_x3/model_best.pth --freeze 1
python train.py --taskName "MVC_multi_v2_50" --instance_dir ../instance/MVC_BA5_6000/train/ --result_dir datasets/MVC_BA5_6000/solution/ --pretrainModel ../multitask/pretrain/MVC_v2/model_best.pth --freeze 1 --fracdata 0.25