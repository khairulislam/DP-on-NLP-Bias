#!/bin/bash -l
# --- this job will be run on any available node
#SBATCH --job-name="Prepare ucberkeley"
#SBATCH --error="prepare-ucberkeley.err"
#SBATCH --output="prepare-ucberkeley.output"
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

EXPERIMENT = "../experiment"
# Prepare train, test, validation files for three different random seeds
python ../preprocess.py --seed 2022 --path $EXPERIMENT --run 1
python ../preprocess.py --seed 42 --path $EXPERIMENT --run 2
python ../preprocess.py --seed 888 --path $EXPERIMENT --run 3

MODEL = "bert-base-uncased" # "distilbert-base-uncased"
python ../../tokenizer.py --model $MODEL --path "$EXPERIMENT/run 1"
python ../../tokenizer.py --model $MODEL --path "$EXPERIMENT/run 2"
python ../../tokenizer.py --model $MODEL --path "$EXPERIMENT/run 3"