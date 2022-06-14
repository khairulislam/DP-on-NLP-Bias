from tqdm.notebook import tqdm
import torch, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from transformers import AutoModelForSequenceClassification

class TrainUtil:
    def __init__(self, id_column, target, device, disable_progress=False):
        self.sigmoid = torch.nn.Sigmoid()
        self.device = device
        self.target = target
        self.id_column = id_column
        self.disable_progress = disable_progress

    def evaluate(self, model, test_dataloader, epoch, data_type='Test'):    
        model.eval()

        losses, total_labels = [], []
        total_probs = torch.tensor([], dtype=torch.float32)
        progress_bar = tqdm(
            range(len(test_dataloader)), desc=f'Epoch {epoch} ({data_type})', 
            disable= self.disable_progress
        )
        
        for batch in test_dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                
            loss = outputs[0]
            
            probs = self.sigmoid(outputs.logits.detach().cpu())[:, 1]
            labels = inputs[self.target].detach().cpu().numpy()
            
            losses.append(loss.item())
            total_probs = torch.cat((total_probs, probs), dim=0)
            total_labels.extend(labels)
            
            progress_bar.update(1)
            
            progress_bar.set_postfix(
                loss=np.round(np.mean(losses), 4), 
                f1=np.round(f1_score(total_labels, total_probs>=0.5), 4)
            )
        
        model.train()
        test_result = TrainUtil.calculate_result(total_labels, total_probs)
        return np.mean(losses), test_result, total_probs

    def train(self, model, train_dataloader, optimizer, epoch):
        model.train()
        
        losses, total_labels = [], []
        total_probs = torch.tensor([], dtype=torch.float32)
        progress_bar = tqdm(
            range(len(train_dataloader)), desc=f'Epoch {epoch} (Train)', 
            disable=self.disable_progress
        )

        for _, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs = {k: v.to(self.device) for k, v in data.items()}
            outputs = model(**inputs) # output = loss, logits, hidden_states, attentions

            # targets = data[target].to(device, dtype = torch.long)
            # loss = loss_function(outputs.logits, targets)
            loss = outputs[0]

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
            probs = self.sigmoid(outputs.logits.detach().cpu())[:, 1]
            labels = inputs[self.target].detach().cpu().numpy()
            
            total_probs = torch.cat((total_probs, probs), dim=0)
            total_labels.extend(labels)

            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=np.round(np.mean(losses), 4), 
                f1=np.round(f1_score(total_labels, total_probs>=0.5), 4)
            )


        train_loss = np.mean(losses)
        train_result = TrainUtil.calculate_result(np.array(total_labels), np.array(total_probs))

        return train_loss, train_result, total_probs

    def dp_train(self, model, optimizer, epoch, memory_safe_data_loader):
        losses, total_labels = [], []
        total_probs = torch.tensor([], dtype=torch.float32)

        progress_bar = tqdm(
            range(len(memory_safe_data_loader)), 
            desc=f'Epoch {epoch} (Train)', disable=self.disable_progress
            )

        for _, data in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()

            inputs = {k: v.to(self.device) for k, v in data.items()}
            outputs = model(**inputs) # output = loss, logits, hidden_states, attentions

            # loss = loss_function(outputs.logits, targets)
            loss = outputs[0]

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
            probs = self.sigmoid(outputs.logits.detach().cpu())[:, 1]
            labels = inputs[self.target].detach().cpu().numpy()
            
            total_probs = torch.cat((total_probs, probs), dim=0)
            total_labels.extend(labels)

            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=np.round(np.mean(losses), 4), 
                f1=np.round(f1_score(total_labels, total_probs>=0.5), 4)
            )

        train_loss = np.mean(losses)
        train_result = TrainUtil.calculate_result(np.array(total_labels), np.array(total_probs))

        return train_loss, train_result, total_probs

    def dump_results(self, result_dir, train_probs, train_all_tokenized, 
        val_probs, validation_all_tokenized, test_probs, test_all_tokenized
    ):
        train_df = pd.DataFrame({
            'id':train_all_tokenized[self.id_column], 'labels':train_all_tokenized[self.target], 
            'probs': train_probs, 'split':['train']* len(train_all_tokenized)
        })
        val_df = pd.DataFrame({
            'id':validation_all_tokenized[self.id_column], 'labels':validation_all_tokenized[self.target], 
            'probs': val_probs, 'split':['validation']* len(validation_all_tokenized)
        })
        test_df = pd.DataFrame({
            'id':test_all_tokenized[self.id_column], 'labels':test_all_tokenized[self.target], 
            'probs': test_probs, 'split':['test']* len(test_all_tokenized)
        })

        total_df = pd.concat([train_df, val_df, test_df],ignore_index=True)
        total_df.to_csv(os.path.join(result_dir, 'results.csv'), index=False)
    
    @staticmethod
    def calculate_result(labels, probs, threshold=0.5):
        preds = np.where(probs >= threshold, 1, 0)
        return {
            'accuracy': np.round(accuracy_score(labels, preds), 4),
            'f1': np.round(f1_score(labels, preds), 4),
            'auc': np.round(roc_auc_score(labels, probs), 4)
        }
    
    @staticmethod
    def load_pretrained_model(model_name, num_labels):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if model_name.startswith('bert'):
            trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
        elif model_name.startswith('distilbert'):
            trainable_layers = [model.distilbert.transformer.layer[-1], model.pre_classifier, model.classifier]
        else:
            print('Warning, trainable layers are not tuned for this model !')
            trainable_layers = [model.classifier]

        total_params = 0
        trainable_params = 0

        for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        print(f"Total parameters count: {total_params}")
        print(f"Trainable parameters count: {trainable_params}, percent {(trainable_params * 100 / total_params):0.3f}")

        return model

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    @staticmethod
    def save_model(model, optimizer, lr_scheduler, epoch, filepath='model.pt'):
        """
        Function to save the trained model to disk.
        """
        torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict':lr_scheduler.state_dict()
        }, 
        filepath
        )

    @staticmethod
    def load_model(model, optimizer, lr_scheduler, device, filepath='model.pt'):
        """
        Function to load the trained model from disk.
        """
        checkpoint = torch.load(filepath, map_location=device) # Choose whatever GPU device number you want  
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded best model from epoch {epoch}')
        
        return model, optimizer, lr_scheduler, epoch

# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
# Custom early stop, save and load models
# The library methods work well with trainer or fit method
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('Early stopping..')
                self.early_stop = True

# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
class ModelCheckPoint:
    """
    Class to save the best model while training. If the current epoch's 
    loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_loss=float('inf'), filepath='best_model.pt'):
        self.best_loss = best_loss
        self.filepath = filepath
        
    def __call__(self, model, optimizer, lr_scheduler, epoch, current_loss):
        if current_loss >= self.best_loss:
            return
        print(f"\nLoss improved from {self.best_loss:.3f} to {current_loss:.3f}. Saving model.")
        self.best_loss = current_loss
        TrainUtil.save_model(model, optimizer, lr_scheduler, epoch, self.filepath)