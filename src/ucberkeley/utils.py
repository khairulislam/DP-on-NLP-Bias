import os, torch, random, pickle
import numpy as np

def dictionary(config) -> dict:
    # exclude hidden variables
    config_dict =config.__dict__.copy()
    keys = list(config_dict.keys())
    for key in keys:
        if key.startswith('__'):
            del config_dict[key]

    return config_dict

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_tokenized(tokenizer_root):
    with open(os.path.join(tokenizer_root, 'train.pkl'), 'rb') as input_file:
        train_tokenized = pickle.load(input_file)
        input_file.close()
    
    with open(os.path.join(tokenizer_root, 'validation.pkl'), 'rb') as input_file:
        validation_tokenized = pickle.load(input_file)
        input_file.close()
        
    with open(os.path.join(tokenizer_root, 'test.pkl'), 'rb') as input_file:
        test_tokenized = pickle.load(input_file)
        input_file.close()

    return train_tokenized, validation_tokenized, test_tokenized