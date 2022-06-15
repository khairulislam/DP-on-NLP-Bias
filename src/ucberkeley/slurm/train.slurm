#!/bin/bash -l
#SBATCH --job-name="Train ucberkeley"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --array=1, 2, 3%1 # will run at most 1 at a time
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

<<<<<<< HEAD
EXPERIMENT = "../experiment"
MODEL = "bert-base-uncased"
python ../../train.py --path "$EXPERIMENT/run $SLURM_ARRAY_TASK_ID/$MODEL"
=======
cd ~/Git/Data-Privacy-Project/src/ucberkeley/
python ../train.py --path "experiment/run $SLURM_ARRAY_TASK_ID/bert-base-uncased"
>>>>>>> main
