import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from dataclasses import dataclass

@dataclass
class Config:
    model_name = 'bert-base-uncased'
    dataset_name = 'ucberkeley-dlab/measuring-hate-speech'
    text_column = 'text'
    # if the raw id column is string, replace that with an integer index during preprocessing 
    id_column = 'comment_id'

    # target in raw dataset. However, it will be renamed to `labels` here to facilitate training setup
    raw_target_column = 'hatespeech'
    target_column = 'labels'
    
    # If needs to be splitted into train test validation set
    need_to_split = False
    # if need_to_split is True, test and validation data with each be 50% of this amount
    test_size = 0.3
    max_seq_length = 128
    seed = 2022

global_seed = Config.seed

# need to install on kaggle or colab
# !pip install datasets
import datasets
dataset = datasets.load_dataset(Config.dataset_name)

text_column = Config.text_column
target_column = Config.target_column
id_column = Config.id_column

# If the dataset is not already splitted into train-test-validation format do that here
need_to_split = False
if need_to_split:
    from sklearn.model_selection import train_test_split
    df = dataset.to_pandas() # Convert dataset to pandas for easier processing

    x_train, x_val, y_train, y_val = train_test_split(
        df.drop(columns=target_column),
        df[target_column],
        test_size=Config.test_size,
        random_state=global_seed
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val,
        y_val,
        test_size=0.5,
        random_state=global_seed
    )

    x_train[target_column] = y_train
    x_val[target_column] = y_val
    x_test[target_column] = y_test

    train = x_train.reset_index(drop=True)
    validation = x_val.reset_index(drop=True)
    test = x_test.reset_index(drop=True)
else:
    # Convert dataset to pandas for easier processing
    train  = dataset['train'].to_pandas()
    test  = dataset['test'].to_pandas()
    validation  = dataset['validation'].to_pandas()

# if you need to drop dulicates or aggregate on data, do those preprocessings here
class Processor:
    @staticmethod
    def process(df):
        return df
train = Processor.process(train)
test = Processor.process(test)
validation = Processor.process(validation)

# Whatever the target column is, it needs to be renamed to `labels` for pytorch train
train.rename({target_column: 'labels'}, axis=1, inplace=True)
test.rename({target_column: 'labels'}, axis=1, inplace=True)
validation.rename({target_column: 'labels'}, axis=1, inplace=True)
target_column = 'labels'
print(f'Target column has been changed from {target_column} to labels')

# Drop unnecessary columns
final_columns = [id_column, text_column, target_column]
train = train[final_columns]
test = test[final_columns]
validation = validation[final_columns]

# Dump dataframe format for future evaluation if necessary
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
validation.to_csv('validation.csv', index=False)

# Check length distribution of the text column to decide max_seg_length
lengths = []
for text in train[text_column].values:
    lengths.append(len(text.split(' ')))
    
lengths = pd.DataFrame(lengths)
lengths.describe()

# Check target column distribution
def value_count(df, value):
    counts = df[value].value_counts().reset_index()
    counts.columns = ['Value', 'Count']
    counts['Count(%)'] = counts['Count'] * 100 / counts['Count'].sum()
    print(counts, '\n')

print('Train dataset')
value_count(train, target_column)

print('Validation dataset')
value_count(validation, target_column)

print('Test dataset')
value_count(test, target_column)

# need to install on google colab
# pip install transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    Config.model_name,
    do_lower_case=True,
)

# create dataset from pandas dataframe
train_dataset = datasets.Dataset.from_pandas(train)
val_dataset = datasets.Dataset.from_pandas(validation)
test_dataset = datasets.Dataset.from_pandas(test)

def tokenize_function(examples):
    return tokenizer(list(examples[text_column]), padding="max_length", max_length=Config.max_seq_length, truncation=True)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)
print(train_tokenized.column_names, val_tokenized.column_names, test_tokenized.column_names)

# https://huggingface.co/docs/datasets/access
# drop string columns because they cause error during training phase

train_tokenized = train_tokenized.remove_columns([text_column])
train_tokenized.set_format("torch")

val_tokenized = val_tokenized.remove_columns([text_column])
val_tokenized.set_format("torch")

test_tokenized = test_tokenized.remove_columns([text_column])
test_tokenized.set_format("torch")

# dump data as pickle
import pickle

with open('train.pkl', 'wb') as output:
    pickle.dump(train_tokenized, output, pickle.HIGHEST_PROTOCOL)
    output.close()
    
with open('validation.pkl', 'wb') as output:
    pickle.dump(val_tokenized, output, pickle.HIGHEST_PROTOCOL)
    output.close()
    
with open('test.pkl', 'wb') as output:
    pickle.dump(test_tokenized, output, pickle.HIGHEST_PROTOCOL)
    output.close()