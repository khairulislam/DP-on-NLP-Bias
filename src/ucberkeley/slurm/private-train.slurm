#!/bin/bash -l
#SBATCH --job-name="Private train ucberkeley"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#---SBATCH --array=3,6,9
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

# python ../../private_train.py --path "../experiment/run 2/bert-base-uncased" --epsilon $SLURM_ARRAY_TASK_ID
python ../../private_train.py --path "../experiment/run 3/bert-base-uncased" --epsilon 1.0
