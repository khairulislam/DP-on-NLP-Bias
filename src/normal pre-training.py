# Need to install on kaggle and google colab
# !pip install datasets
import datasets

# Need to install on google colab
# !pip install transformers
import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import gc
from dataclasses import dataclass

pd.set_option('display.max_columns', None)

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Set config
from dataclasses import dataclass
@dataclass
class Config:
    # train config
    model_name = 'bert-base-uncased'
    batch_size = 64
    learning_rate = 1e-4
    epochs = 20
    num_labels = 2

    dataset_name = 'social_bias_frames'
    text_column = 'post'

    # if the raw id column is string, replace that with an integer index during preprocessing 
    raw_id_column = 'HITId'
    id_column = 'index'

    # target in raw dataset is offensiveYN. However, it will be renamed to `labels` here to facilitate training setup
    raw_target_column = 'offensiveYN'
    target_column = 'labels'
    
    # If needs to be splitted into train test validation set
    need_to_split = False
    # test and validation data with each be 50% of this amount
    test_size = 0.3
    max_seq_length = 128
    seed = 2022

# Set seed
import random

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

global_seed = Config.seed
seed_torch(global_seed)

# Get device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Load tokenized data
text = Config.text_column
target = Config.target_column
root = '/kaggle/input/tokenize-social-bias-using-bert/'

import pickle
    
with open(root + 'train.pkl', 'rb') as input_file:
    train_tokenized = pickle.load(input_file)
    input_file.close()
    
with open(root + 'validation.pkl', 'rb') as input_file:
    validation_tokenized = pickle.load(input_file)
    input_file.close()
    
with open(root + 'test.pkl', 'rb') as input_file:
    test_tokenized = pickle.load(input_file)
    input_file.close()

print(train_tokenized)

# Remove id column from the data to be batched
id_column = Config.id_column

train_tokenized = train_tokenized.remove_columns(id_column)
test_tokenized = test_tokenized.remove_columns(id_column)
validation_tokenized = validation_tokenized.remove_columns(id_column)

# Training phase
# Data loader
BATCH_SIZE = Config.batch_size

train_dataloader = DataLoader(train_tokenized, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_tokenized, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_tokenized, batch_size=BATCH_SIZE)

from train_utils import TrainUtil, EarlyStopping, ModelCheckPoint

num_labels = Config.num_labels
model_name = Config.model_name
train_util = TrainUtil(Config.id_column, Config.target_column, device)

model = TrainUtil.load_pretrained_model(model_name, num_labels)

# Define optimizer
LEARNING_RATE = Config.learning_rate
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
EPOCHS = Config.epochs

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True) 

result_dir = ''
best_model_path = result_dir + 'model.pt'
if result_dir != '':
    os.makedirs(result_dir, exist_ok=True)

check_point = ModelCheckPoint(filepath=best_model_path)
early_stopping = EarlyStopping(patience=3, min_delta=0)

start_epoch = 1
# load a previous model if there is any
# model, optimizer, lr_scheduler, start_epoch = load_model(model, optimizer, lr_scheduler, device, filepath=best_model_path)
model = model.to(device)

for epoch in range(start_epoch, EPOCHS+1):
    gc.collect()
    
    train_loss, train_result, train_probs = train_util.train(model, train_dataloader, optimizer, epoch)
    val_loss, val_result, val_probs = train_util.evaluate(model, validation_dataloader, epoch, 'Validation')

    print(
      f"Epoch: {epoch} | "
      f"Train loss: {train_loss:.3f} | "
      f"Train result: {train_result} |\n"
      f"Validation loss: {val_loss:.3f} | "
      f"Validation result: {val_result} | "
    )
    
    loss = -val_result['f1']
    lr_scheduler.step(loss)
    check_point(model, optimizer, lr_scheduler, epoch, loss)
    
    early_stopping(loss)
    if early_stopping.early_stop:
        break
    print()


# load the best model
model, _, _, best_epoch = TrainUtil.load_model(model, optimizer, lr_scheduler, device, filepath=best_model_path)

train_loss, train_result, train_probs = train_util.evaluate(model, train_dataloader, best_epoch, 'Train')
# no need to reevaluate if the validation set if the last model is the best one
if best_epoch != epoch:
    val_loss, val_result, val_probs = train_util.evaluate(model, validation_dataloader, best_epoch, 'Validation')
test_loss, test_result, test_probs = train_util.evaluate(model, test_dataloader, best_epoch, 'Test')

# load the original tokenized files, since we removed the id columns earlier
# and id columns are needed for the result dumping part
with open(root + 'train.pkl', 'rb') as input_file:
    train_tokenized = pickle.load(input_file)
    input_file.close()
    
with open(root + 'validation.pkl', 'rb') as input_file:
    validation_tokenized = pickle.load(input_file)
    input_file.close()
    
with open(root + 'test.pkl', 'rb') as input_file:
    test_tokenized = pickle.load(input_file)
    input_file.close()

# Save the results
train_util.dump_results(
    result_dir,train_probs, train_tokenized, 
    val_probs, validation_tokenized, test_probs, test_tokenized
)

# Save config
import json

config_dict = dict(Config.__dict__)
# exclude hidden variables
keys = list(config_dict.keys())
for key in keys:
    if key.startswith('__'):
        del config_dict[key]
        
with open(os.path.join(result_dir, 'config.json'), 'w') as output:
    json.dump(config_dict, output, indent=4)
