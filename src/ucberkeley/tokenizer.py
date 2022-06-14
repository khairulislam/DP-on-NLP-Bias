"""
python tokenizer.py --model "bert-base-uncased" --path "experiment/run 1"
python tokenizer.py --model "bert-base-uncased" --path "experiment/run 2"
python tokenizer.py --model "bert-base-uncased" --path "experiment/run 3"
"""
import pandas as pd
import argparse
# need to install on kaggle or colab
# !pip install datasets
import datasets, pickle
# need to install on google colab
# pip install transformers
from transformers import AutoTokenizer
import json, os

from utils import dictionary
from dataclasses import dataclass

@dataclass
class Config:
    max_seq_length = 128

def get_arguments():
    parser = argparse.ArgumentParser(description='Tokenize ucberkeley', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model',help='Model whose tokenizer and pretrained version will be used', 
        type=str, default='bert-base-uncased'
    )
    parser.add_argument(
        '-p', '--path',help="Experiment file's directory.", 
        type=str, default=None
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    experiment_folder = args.path
    Config.model_name = args.model

    with open(os.path.join(experiment_folder, 'config.json')) as inputfile:
        configDict = json.load(inputfile)

    output_folder = os.path.join(args.path, args.model)
    if not os.path.exists(output_folder):
        print(f'Creating output folder {args.path}')
        os.makedirs(output_folder, exist_ok=True)

    text_column = configDict['text_column']
    target_column = configDict['target_column']
    id_column = configDict['id_column']

    # load input files
    train = pd.read_csv(os.path.join(experiment_folder, 'train.csv'))
    validation = pd.read_csv(os.path.join(experiment_folder, 'validation.csv'))
    test = pd.read_csv(os.path.join(experiment_folder, 'test.csv'))

    # Drop unnecessary columns
    final_columns = [id_column, text_column, target_column]

    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        do_lower_case=True,
    )
    
    def tokenize_function(examples):
        return tokenizer(list(examples[text_column]), padding="max_length", max_length=Config.max_seq_length, truncation=True)

    def process(df, split='train'):
        df = df[final_columns]

        # create dataset from pandas dataframe
        dataset = datasets.Dataset.from_pandas(df)
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized = tokenized.remove_columns([text_column])
        tokenized.set_format("torch")

        with open(os.path.join(output_folder, f'{split}.pkl'), 'wb') as output:
            pickle.dump(tokenized, output, pickle.HIGHEST_PROTOCOL)
            output.close()

    process(train, 'train')
    process(validation, 'validation')
    process(test, 'test')

    with open(os.path.join(output_folder, 'config.json'), 'w') as output:
        json.dump(configDict | dictionary(Config), output, indent=4)

if __name__ == '__main__':
    main()