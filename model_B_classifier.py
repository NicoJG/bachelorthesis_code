import sys
from copy import deepcopy
import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.data_handling import DataIteratorByEvents

# Define the model
# similar to https://gitlab.cern.ch/nnolte/DeepSetTagging/-/blob/cf286579b7db4ff21341eb4b06fc8f726168a8c9/model.py
# although this is a PyTorch model
# it is meant to be used with numpy arrays
# and the sklearn-like functions fit, predict, predict_proba are available
class DeepSetModel(nn.Module):
    def __init__(self, n_features, optimizer=torch.optim.Adam, optimizer_kwargs={}, loss=nn.BCELoss(), scaler=StandardScaler()):
        super(DeepSetModel, self).__init__()
        
        assert issubclass(optimizer, torch.optim.Optimizer), "The optimizer has to come from 'torch.optim'!"
        assert callable(loss), "The loss has to be callable!"
        
        self.n_features = n_features
        self.loss = loss
        self.scaler = scaler
        
        self.train_history = None
        self.is_fitted = False
        
        self.classes_ = [0., 1.]
        
        # DeepSet Structure:
        
        # Neural Network for the tracks:
        self.phi_stack = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        # Sum up all outputs of the phi_stack for each event
        # done in the forward function
        
        # Neural Network for the events
        self.rho_stack = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def forward(self, X):
        # X must have shape (tracks, features+1)
        # the first column of X has to be the event_id for each track
        event_ids = X[:,0]
        X = X[:,1:]
        
        # get an event_index_by_track array (starts at 0 and is counting up for each new event)
        idxs = (event_ids != event_ids.roll(1,0)).cumsum(dim=0, dtype=int) - 1
        
        if idxs[-1] > len(event_ids.unique()):
            # this is for the Permutation Importance (where also the first feature is permutated (event_ids))
            print("WARNING: Same event ids can not be seperated in X! Producing an error prediction!")
            return -1*torch.ones(len(event_ids.unique())).float()
        
        # Pass X through the DeepSet:
        
        X = self.phi_stack(X)
        
        # sum up the latent features of all tracks per event
        temp = torch.zeros(idxs[-1]+1, X.shape[1]).to(X.device)
        X = temp.index_add(dim=0, index=idxs, source=X)
    
        y = self.rho_stack(X)
        
        return y

    def _train_loop(self, dataiterator, pbar=None, show_batch_eval=False):
        loss_sum, error_count, event_count = 0, 0, 0
        
        self.train()
        
        for batch_i, (X, y, event_ids) in enumerate(dataiterator):
            y_pred = self(X)
            loss = self.loss(y_pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_loss = loss.item()
            batch_error = ((y_pred > 0.5) != y).sum().item()

            loss_sum += batch_loss
            error_count += batch_error
            event_count += len(y)
            
            if pbar is not None:
                pbar.update()
                
            if show_batch_eval:
                print_file = sys.stdout if pbar is None else sys.stderr
                print(f"Batch {batch_i:03d}/{len(dataiterator)}: {batch_loss = :.4f} ; {batch_error = :.4f}", file=print_file)
            
        loss = loss_sum / len(dataiterator)
        error = error_count / event_count
        
        self.eval()
        
        return loss, error
    
    def _test_loop(self, dataiterator, pbar=None):
        loss_sum, error_count, event_count = 0, 0, 0
        
        with torch.no_grad():
            for X, y, event_ids in dataiterator:
                y_pred = self(X)
                loss_sum += self.loss(y_pred, y).item()
                error_count += ((y_pred > 0.5) != y).sum().item()
                event_count += len(y)
                if pbar is not None:
                    pbar.update()
            
        loss = loss_sum / len(dataiterator)
        error = error_count / event_count

        return loss, error
    
    def _scale_X(self, X):
        if isinstance(X, torch.Tensor):
            is_tensor = True
            X = X.numpy()
        else:
            is_tensor = False
            
        if self.scaler is not None:
            temp = self.scaler.transform(X[:,1:])
            X = np.concatenate([X[:,0][:,np.newaxis], temp], axis=1)
            
        if is_tensor:
            return torch.from_numpy(X).float()
        else:
            return X
    
    def fit(self, X, y, epochs=1, batch_size=1, 
            X_val=None, y_val=None, 
            device=None, 
            show_epoch_progress=True, show_epoch_eval=False, 
            show_batch_progress=True, show_batch_eval=False,
            early_stopping_epochs=None):
        
        assert not self.is_fitted, "This DeepSet is already fitted."
        
        is_validation_provided = X_val is not None and y_val is not None
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(y_val, np.ndarray):
            y_val = torch.from_numpy(y_val).float()
        
        self.scaler.fit(X[:,1:].numpy())
        X = self._scale_X(X)
        if is_validation_provided:
            X_val = self._scale_X(X_val)
            
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
            if is_validation_provided:
                y_val = y_val.unsqueeze(1)
                
        if device is not None:
            X = X.to(device)
            y = y.to(device)
            if is_validation_provided:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                        
        train_history = {"epochs" : [],
                         "eval_metrics" : ["loss", "error"],
                         "train": {"loss":[],"error":[]},
                         "train_wrong": {"loss":[],"error":[]}}
        
        if is_validation_provided:
            train_history["validation"] = {"loss":[],"error":[]}
            val_iterator = DataIteratorByEvents(X_val, y_val, batch_size=batch_size)
        
        train_iterator = DataIteratorByEvents(X, y, batch_size=batch_size)
        
        is_early_stopping = isinstance(early_stopping_epochs, int)
        
        # save the best model
        best_epoch = -1
        best_val_loss = np.Infinity
        
        if show_epoch_progress:
            epoch_iter = tqdm(range(epochs), desc="Train epochs")
        else:
            epoch_iter = range(epochs)
        
        for epoch_i in epoch_iter:
            if show_batch_progress:
                if epoch_i == 0:
                    pbar = tqdm(total=len(train_iterator), desc="Batches")
                else:
                    pbar.refresh()
                    pbar.reset()
            else:
                pbar = None
                
            train_loss, train_error = self._train_loop(train_iterator, pbar, show_batch_eval)
            
            train_history["epochs"].append(epoch_i)
            train_history["train_wrong"]["loss"].append(train_loss)
            train_history["train_wrong"]["error"].append(train_error)
            
            # Dropout correction
            train_corrected_loss, train_corrected_error = self._test_loop(train_iterator)
            train_history["train"]["loss"].append(train_corrected_loss)
            train_history["train"]["error"].append(train_corrected_error)
            
            if is_validation_provided:
                val_loss, val_error = self._test_loop(val_iterator)
                
                train_history["validation"]["loss"].append(val_loss)
                train_history["validation"]["error"].append(val_error)
                
                if val_loss < best_val_loss:
                    best_epoch = epoch_i
                    best_val_loss = val_loss
                    best_model_state_dict = deepcopy(self.state_dict())
                    best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
                
            if show_epoch_eval and is_validation_provided:
                print_file = sys.stdout if not show_epoch_progress else sys.stderr
                print(f"\nEpoch {epoch_i:03d}/{epochs} (best: {best_epoch}):", file=print_file)
                print(f"train loss: {train_corrected_loss:.4f} ; train error: {train_corrected_error:.4f}", file=print_file)
                print(f"val loss:   {val_loss:.4f} ; val error:   {val_error:.4f}", file=print_file)
                
            if is_validation_provided and is_early_stopping:
                # very basic early stopping
                if epoch_i - best_epoch > early_stopping_epochs:
                    print("Early Stopping...")
                    break
                    
            
        # save only the best model
        if is_validation_provided:
            train_history["best_epoch"] = best_epoch
            self.load_state_dict(best_model_state_dict)
            self.optimizer.load_state_dict(best_optimizer_state_dict)
            
        self.train_history = train_history
        self.is_fitted = True
        
    def decision_function(self, X, is_scaled=False):
        # Output shape: (n_samples,)
        assert self.is_fitted, "This DeepSet is not yet fitted."
        
        if isinstance(X, np.ndarray):
            is_numpy = True
            X = torch.from_numpy(X).float()
        else:
            is_numpy = False
        
        if not is_scaled:
            X = self._scale_X(X)
            
        y = self(X).squeeze()
        
        if is_numpy:
            return y.detach().numpy()
        else:
            return y
        
    def predict_proba(self, X, is_scaled=False):
        # output shape: (n_samples, n_classes)
        y = self.decision_function(X, is_scaled=is_scaled)
        
        if isinstance(y, np.ndarray):
            y = np.column_stack([1-y, y])
        else:
            y = torch.column_stack([1-y, y])
            
        return y
        
    
    def predict(self, X, is_scaled=False):
        # output shape: (n_samples,)
        y = self.decision_function(X, is_scaled=is_scaled)
        
        if isinstance(y, np.ndarray):
            return (y > 0.5).astype(int)
        else:
            return (y > 0.5).int()
        
