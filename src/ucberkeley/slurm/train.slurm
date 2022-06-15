#!/bin/bash -l
#SBATCH --job-name="Private train ucberkeley"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --array=1, 2, 3
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

cd ~/Git/Data-Privacy-Project/src/ucberkeley/
python ../train.py --path "experiment/run $SLURM_ARRAY_TASK_ID/bert-base-uncased"