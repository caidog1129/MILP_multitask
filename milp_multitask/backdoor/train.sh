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
python train.py --taskName "CA_175_850_50" --instance_dir ../instance/CA_175_850/train/ --result_dir datasets/CA_175_850/SOL/ --fracdata 0.25
# python train.py --taskName "CA_175_850" --instance_dir ../instance/CA_175_850/train/ --result_dir datasets/CA_175_850/SOL/ --pretrainModel ../multitask/pretrain/CA/model_best.pth --freeze 1
# python train.py --taskName "CA_175_850_multi_v3_50" --instance_dir ../instance/CA_175_850/train/ --result_dir datasets/CA_175_850/SOL/ --pretrainModel ../multitask/pretrain/CA_v3/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "INDSET_BA4_1250_50" --instance_dir ../instance/INDSET_BA4_1250/train/ --result_dir datasets/INDSET_BA4_1250/SOL/ --fracdata 0.25
# python train.py --taskName "INDSET_BA4_1250_multi" --instance_dir ../instance/INDSET_BA4_1250/train/ --result_dir datasets/INDSET_BA4_1250/SOL/ --pretrainModel ../multitask/pretrain/IS/model_best.pth --freeze 1
# python train.py --taskName "INDSET_BA4_1250_multi_v3_50" --instance_dir ../instance/INDSET_BA4_1250/train/ --result_dir datasets/INDSET_BA4_1250/SOL/ --pretrainModel ../multitask/pretrain/IS_v3/model_best.pth --freeze 1 --fracdata 0.25

# python train.py --taskName "MVC_BA5_1500_50" --instance_dir ../instance/MVC_BA5_1500/train/ --result_dir datasets/MVC_BA5_1500/SOL/ --fracdata 0.25
# python train.py --taskName "MVC_BA5_1500_multi" --instance_dir ../instance/MVC_BA5_1500/train/ --result_dir datasets/MVC_BA5_1500/SOL/ --pretrainModel ../multitask/pretrain/MVC/model_best.pth --freeze 1
# python train.py --taskName "MVC_BA5_1500_multi_v3_50" --instance_dir ../instance/MVC_BA5_1500/train/ --result_dir datasets/MVC_BA5_1500/SOL/ --pretrainModel ../multitask/pretrain/MVC_v3/model_best.pth --freeze 1 --fracdata 0.25