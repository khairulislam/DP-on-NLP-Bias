# Need to install on kaggle and google colab
# !pip install datasets
import datasets


# !pip install opacus
import opacaus
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Need to install on google colab
# !pip install transformers

from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, Conv1D
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import gc

pd.set_option('display.max_columns', None)

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Set config
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
    
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    num_labels = 2
    
    # Private training config
    delta_list = [5e-2, 1e-3, 1e-6]
    noise_multiplier = 0.45
    max_grad_norm = 1
    max_physical_batch_size = 32
    target_epsilon = 9.0

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

# Training phase
# Data loader
BATCH_SIZE = Config.batch_size
# Remove id column from the data to be batched
id_column = Config.id_column

train_dataloader = DataLoader(train_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE*5)
test_dataloader = DataLoader(test_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE*5)

# on Kaggle add the utility script from File->Add utility script
from train_utils import TrainUtil, ModelCheckPoint, EarlyStopping
num_labels = Config.num_labels
model_name = Config.model_name
model = TrainUtil.load_pretrained_model(model_name, num_labels)

# Define optimizer
LEARNING_RATE = Config.learning_rate
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
EPOCHS = Config.epochs

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True) 

result_dir = ''
best_model_path = os.path.join(result_dir, 'model.pt')
if result_dir != '':
    os.makedirs(result_dir, exist_ok=True)

check_point = ModelCheckPoint(filepath=best_model_path)
early_stopping = EarlyStopping(patience=3, min_delta=0)
train_util = TrainUtil(Config.id_column, Config.target_column, device)

# Privacy engine
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

# model, optimizer, train_dataloader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_dataloader,
#     noise_multiplier=Config.noise_multiplier,
#     max_grad_norm=Config.max_grad_norm,
#     poisson_sampling=False,
# )

model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    target_delta=Config.delta_list[-1],
    target_epsilon=Config.target_epsilon, 
    epochs=EPOCHS,
    max_grad_norm=Config.max_grad_norm,
)

# Train loop
# load a previous model if there is any
# model, optimizer, lr_scheduler, start_epoch = load_model(model, optimizer, lr_scheduler, device, filepath=best_model_path)
model = model.to(device)

for epoch in range(1, EPOCHS+1):
    gc.collect()
    
    with BatchMemoryManager(
        data_loader=train_dataloader, 
        max_physical_batch_size=Config.max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        train_loss, train_result, train_probs = train_util.dp_train(
            model, optimizer, epoch, memory_safe_data_loader
        )
    val_loss, val_result, val_probs = train_util.evaluate(
        model, validation_dataloader, epoch, 'Validation'
    )

    epsilons = []
    for delta in Config.delta_list:
        epsilons.append(privacy_engine.get_epsilon(delta))

    print(
      f"Epoch: {epoch} | "
      f"ɛ: {np.round(epsilons, 2)} |"
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

# make_private_with_epsilon function creates inconsistent train dataloader size
train_dataloader = DataLoader(train_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE*5)
train_loss, train_result, train_probs = train_util.evaluate(model, train_dataloader, best_epoch, 'Train')
# no need to reevaluate if the validation set if the last model is the best one
if best_epoch != epoch:
    val_loss, val_result, val_probs = train_util.evaluate(model, validation_dataloader, best_epoch, 'Validation')
test_loss, test_result, test_probs = train_util.evaluate(model, test_dataloader, best_epoch, 'Test')

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
