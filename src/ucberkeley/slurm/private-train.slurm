#!/bin/bash -l
#SBATCH --job-name="Private train ucberkeley"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --array=3,6,9
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

cd ~/Git/Data-Privacy-Project/src/ucberkeley/
python private-train.py --path "experiment/run 1/bert-base-uncased" --epsilon $SLURM_ARRAY_TASK_ID