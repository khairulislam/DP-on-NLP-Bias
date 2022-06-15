#!/bin/bash

DATASET = "ucberkeley" # "jigsaw"

cd $DATASET
python preprocess.py --seed 2022 --path "experiment" --run 1
python preprocess.py --seed 42 --path "experiment" --run 2
python preprocess.py --seed 888 --path "experiment" --run 3
cd ..

for RUN in 1 2 3
do
    python ../tokenizer.py --model "bert-base-uncased" --path "$DATASET/experiment/run $RUN"

    python ../tokenizer.py --model "distilbert-base-uncased" --path "$DATASET/experiment/run $RUN"

    python ../train.py --path "$DATASET/experiment/run $RUN/bert-base-uncased"
    python ../train.py --path "$DATASET/experiment/run $RUN/distilbert-base-uncased"

    for EPSILON in 3 6 9
    do 
        python ../private-train.py --path "$DATASET/experiment/run $RUN/bert-base-uncased" --epsilon $EPSILON
        python ../private-train.py --path "$DATASET/experiment/run $RUN/distilbert-base-uncased" --epsilon $EPSILON
    done
do
