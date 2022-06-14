"""
python preprocess.py --seed 2022 --path "experiment" --run 1
python preprocess.py --seed 42 --path "experiment" --run 2
python preprocess.py --seed 888 --path "experiment" --run 3
"""
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
# need to install on kaggle or colab
# !pip install datasets
import datasets
import json, os, sys

from utils import dictionary
from dataclasses import dataclass

@dataclass
class Config:
    dataset_name = 'ucberkeley-dlab/measuring-hate-speech'

    text_column = 'text'
    # if the raw id column is string, replace that with an integer index during preprocessing
    id_column = 'comment_id'

    # target in raw dataset. However, it will be renamed to `labels` here to facilitate training setup
    raw_target_column = 'hatespeech'
    target_column = 'labels'
    
    test_size = 0.15
    validation_size = 0.15

# Check target column distribution
def value_count(df, value):
    counts = df[value].value_counts().reset_index()
    counts.columns = ['Value', 'Count']
    counts['Count(%)'] = counts['Count'] * 100 / counts['Count'].sum()
    print(counts, '\n')

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocess ucberkeley', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-s', '--seed',help='Seed for random methods', 
        type=int, default=None
    )

    parser.add_argument(
        '-p', '--path',help='Folder location to save and load files for this experiment.', 
        type=str, default='experiment'
    )

    parser.add_argument(
        '-r', '--run',help='The serial number for this experiment', 
        type=int, default=1
    )

    return parser.parse_args()

def main():
    args = get_arguments()
    Config.seed = args.seed

    experiment_folder = os.path.join(args.path, f'run {args.run}')

    if not os.path.exists(experiment_folder):
        print(f'Creating output folder {args.path}')
        os.makedirs(experiment_folder, exist_ok=True)

    original_stdout = sys.stdout
    output_file = open(os.path.join(experiment_folder, 'preprocessor-output.txt'), 'w')
    sys.stdout = output_file

    global_seed = Config.seed

    dataset = datasets.load_dataset(Config.dataset_name)
    df = dataset['train'].to_pandas()
    print(f'Dataset shape {df.shape}')

    text_column = Config.text_column
    target_column = Config.target_column
    id_column = Config.id_column

    # The hatespeech column has three values
    # * 0 for positive comments
    # * 1 when not clear
    # * 2 for hate speech

    # This value can differ among annotators for the same comment_id. 
    # But the calculated hate_speech_score will be the same. 
    # For simplicity in experiment we change the dataset into binary classification 
    # by removing examples where annotators are not clear.

    # https://stackoverflow.com/questions/8689795/how-can-i-remove-non-ascii-characters-but-leave-periods-and-spaces
    # Make sure all comment_text values are strings
    df.loc[:, text_column] = df[text_column].astype(str) 

    # Whatever the target column is, it needs to be renamed to `labels` for pytorch train
    print(f'Target column name has been changed from {Config.raw_target_column} to labels')
    df.loc[:, target_column] = df[Config.raw_target_column].astype(int)
    df = df[df[target_column] != 1]
    df.loc[:, target_column] = df[target_column].map({0:0, 2:1})

    """
    Aggregate duplicate rows, since there are multiple annotations for the same comment
    """
    target_identities = [col for col in df.columns if 'target_' in col]
    # https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
    grouped = df.groupby([id_column])[target_identities].agg('mean').reset_index()
    for identity in target_identities:
        grouped[identity] = grouped[identity].apply(lambda x: x >= 0.5)

    dataset_unique = df.drop_duplicates(subset=id_column)[[id_column, text_column, target_column]]
    df = dataset_unique.merge(grouped, on=id_column, how='inner').reset_index(drop=True)
    print(f'Dataset shape after aggregating annotations {df.shape}')

    """
    train test validation split
    """
    x_train, x_val, y_train, y_val = train_test_split(
        df.drop(columns=target_column),
        df[target_column],
        test_size=Config.test_size+Config.validation_size,
        random_state=global_seed
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val,
        y_val,
        test_size=Config.validation_size / (Config.test_size+Config.validation_size),
        random_state=global_seed
    )

    x_train[target_column] = y_train
    x_val[target_column] = y_val
    x_test[target_column] = y_test

    train = x_train.reset_index(drop=True)
    validation = x_val.reset_index(drop=True)
    test = x_test.reset_index(drop=True)

    print('Train dataset')
    value_count(train, target_column)

    print('Validation dataset')
    value_count(validation, target_column)

    print('Test dataset')
    value_count(test, target_column)

    # Dump dataframe format for future evaluation if necessary
    train.to_csv(os.path.join(experiment_folder, 'train.csv'), index=False)
    test.to_csv(os.path.join(experiment_folder, 'test.csv'), index=False)
    validation.to_csv(os.path.join(experiment_folder, 'validation.csv'), index=False)

    with open(os.path.join(experiment_folder, 'config.json'), 'w') as output:
        json.dump(dictionary(Config), output, indent=4)

    sys.stdout = original_stdout
    output_file.close()

if __name__ == '__main__':
    main()