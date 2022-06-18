"""
After running these following lines, copy the train, test, validation
csv files manually into the corresponding results/run folder

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

sys.path.append('..')
from train_utils import dictionary, value_count
from dataclasses import dataclass

@dataclass
class Config:
    dataset_name = 'jigsaw-unintended-bias-in-toxicity-classification'

    text_column = 'comment_text'
    # if the raw id column is string, replace that with an integer index during preprocessing
    id_column = 'id'

    # target in raw dataset. However, it will be renamed to `labels` here to facilitate training setup
    raw_target_column = 'toxicity'
    target_column = 'labels'
    undersample = False
    validation_size = 0.20

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocess jigsaw', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-s', '--seed',help='Seed for random methods', 
        type=int, default=None
    )

    parser.add_argument(
        '-i', '--input',help='Folder location of the all_data.csv file', 
        type=str, default='experiment'
    )

    parser.add_argument(
        '-p', '--path',help='Folder location to save and load files for this experiment', 
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
        print(f'Creating output folder {experiment_folder}')
        os.makedirs(experiment_folder, exist_ok=True)

    original_stdout = sys.stdout
    output_file = open(os.path.join(experiment_folder, 'preprocessor-output.txt'), 'w')
    sys.stdout = output_file

    global_seed = Config.seed

    JIGSAW_PATH = args.input
    df = pd.read_csv(os.path.join(JIGSAW_PATH,'all_data.csv'))
    print(f'Dataset shape {df.shape}')

    text_column = Config.text_column
    target_column = Config.target_column
    id_column = Config.id_column

    df.loc[:, text_column] = df[text_column].astype(str)
    df[text_column].fillna('', inplace=True) 
    df[target_column] = df[Config.raw_target_column]>=0.5

    id_column = Config.id_column
    identities = ['male', 'female', 'transgender','other_gender', 'white', 'black', 'asian', 'latino', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'christian', 'jewish', 'muslim', 'hindu']
    selected_columns = [id_column, text_column, target_column, 'split'] + identities
    df = df[selected_columns]

    # Train validation split
    train = df[df['split']=='train'].reset_index(drop=True)
    test = df[df['split']=='test'].reset_index(drop=True)

    train.drop(columns='split', inplace=True)
    test.drop(columns='split', inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(
        train.drop(columns=target_column),
        train[target_column],
        stratify=train[target_column],
        test_size=Config.validation_size,
        random_state=global_seed
    )

    if Config.undersample:
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=global_seed)
        x_train, y_train = sampler.fit_resample(x_train, y_train)

        x_train[target_column] = y_train
        x_train = x_train.sample(frac=1).reset_index(drop=True)
    else:
        x_train[target_column] = y_train
        train = x_train.reset_index(drop=True)
    
    x_val[target_column] = y_val
    validation = x_val.reset_index(drop=True)

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