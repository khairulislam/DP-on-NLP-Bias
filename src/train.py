"""
python train.py -p "ucberkeley/experiment/run 1/bert-base-uncased"
"""

import argparse
import json, os, gc
import torch
from torch.utils.data import DataLoader

from train_utils import *
from dataclasses import dataclass
@dataclass
class Config:
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    num_labels = 2
    early_stopping = 3

def get_arguments():
    parser = argparse.ArgumentParser(description='Tokenize ucberkeley', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-p', '--path',help="Tokenized file's directory.", 
        type=str, default=None
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    tokenizer_root = args.path
    output_folder = os.path.join(args.path, 'normal')

    if not os.path.exists(output_folder):
        print(f'Creating output folder {output_folder}')
        os.makedirs(output_folder, exist_ok=True)
    
    # tokenizer should have its own config.json
    with open(os.path.join(tokenizer_root, 'config.json')) as inputfile:
        configDict = json.load(inputfile)

    # set seed
    global_seed = configDict['seed']
    seed_torch(global_seed)

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    train_tokenized, validation_tokenized, test_tokenized = get_tokenized(tokenizer_root)

    # Training phase
    # Data loader
    BATCH_SIZE = Config.batch_size
    # Remove id column from the data to be batched
    id_column = configDict['id_column']

    train_dataloader = DataLoader(train_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE*5)
    test_dataloader = DataLoader(test_tokenized.remove_columns(id_column), batch_size=BATCH_SIZE*5)

    num_labels = Config.num_labels
    model_name = configDict['model_name']
    target_column = configDict['target_column']

    train_util = TrainUtil(id_column, target_column, device, disable_progress=False)
    model = TrainUtil.load_pretrained_model(model_name, num_labels)

    # Define optimizer
    LEARNING_RATE = Config.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    EPOCHS = Config.epochs

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True) 

    best_model_path = os.path.join(output_folder, 'model.pt')

    check_point = ModelCheckPoint(filepath=best_model_path)
    early_stopping = EarlyStopping(patience=Config.early_stopping, min_delta=0)

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

    print(f'At best epoch, train result {train_result}\nvalidation result {val_result}\n, test result {test_result}')
    # Save the results
    train_util.dump_results(
        output_folder, train_probs, train_tokenized, 
        val_probs, validation_tokenized, test_probs, test_tokenized
    )
            
    with open(os.path.join(output_folder, 'config.json'), 'w') as output:
        json.dump(configDict | dictionary(Config), output, indent=4)

if __name__ == '__main__':
    main()