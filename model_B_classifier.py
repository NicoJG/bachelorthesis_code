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
    def __init__(self, n_features, n_latent_features, optimizer=torch.optim.Adam, optimizer_kwargs={}, loss=nn.BCELoss(), scaler=StandardScaler()):
        super(DeepSetModel, self).__init__()
        
        assert issubclass(optimizer, torch.optim.Optimizer), "The optimizer has to come from 'torch.optim'!"
        assert callable(loss), "The loss has to be callable!"
        
        self.n_features = n_features
        self.n_latent_features = n_latent_features
        
        self.loss = loss
        
        self.train_history = None
        self.scaler = scaler
        
        self.is_fitted = False
        
        self.classes_ = [0., 1.]
        
        # DeepSet Structur:
        
        # Neural Network for the tracks:
        self.phi_stack = nn.Sequential(
            nn.Linear(n_features, n_features*2),
            nn.ReLU(),
            nn.Linear(n_features*2, n_latent_features),
            nn.ReLU()
        )
        
        # Sum up all outputs of the phi_stack for each event
        # done in the forward function
        
        # Neural Network for the events
        self.rho_stack = nn.Sequential(
            nn.Linear(n_latent_features, n_latent_features*2),
            nn.ReLU(),
            nn.Linear(n_latent_features*2, n_latent_features),
            nn.ReLU(),
            nn.Linear(n_latent_features, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        
    def forward(self, X):
        # X must have shape (tracks, features+1)
        # the first column of X has to be the event_id for each track
        event_ids = X[:,0]
        X = X[:,1:]
        
        X = self.phi_stack(X)
        
        # get an event_index_by_track array
        idxs = torch.zeros_like(event_ids, dtype=int)
        idxs[1:] = torch.cumsum(event_ids[1:] != event_ids[:-1], 0)
        
        if idxs[-1] > len(event_ids.unique()):
            print("WARNING: Same event ids can not be seperated in X! Producing an error prediction!")
            # this is for the Permutation Importance (it also permutates the first feature (event_ids))
            return -1*torch.ones(len(event_ids.unique())).float()
        
        # sum up the latent features of all tracks per event
        temp = torch.zeros(idxs[-1]+1, self.n_latent_features)
        X = temp.index_add(0, idxs, X)
    
        y = self.rho_stack(X)
        
        return y

    def _train_loop(self, dataiterator, pbar=None):
        loss_sum, error_count, event_count = 0, 0, 0
        
        self.train()
        
        for X, y, event_ids in dataiterator:
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
    
    def fit(self, X, y, epochs=1, batch_size=1, verbose=1, X_val=None, y_val=None, device=None):
        assert not self.is_fitted, "This DeepSet is already fitted."
        
        is_validation_provided = X_val is not None and y_val is not None
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val)
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
            X.to(device)
            y.to(device)
            if is_validation_provided:
                X_val.to(device)
                y_val.to(device)
        
        train_history = {"epochs" : [],
                         "eval_metrics" : ["loss", "error"],
                         "train": {"loss":[],"error":[]}}
        
        
        if is_validation_provided:
            train_history["validation"] = {"loss":[],"error":[]}
            val_iterator = DataIteratorByEvents(X_val, y_val, batch_size=batch_size)
        
        train_iterator = DataIteratorByEvents(X, y, batch_size=batch_size)
        
        for epoch_i in tqdm(range(epochs), desc="Train epochs"):
            if epoch_i == 0:
                pbar = tqdm(total=len(train_iterator), desc="Batches")
            else:
                pbar.refresh()
                pbar.reset()
                
            train_loss, train_error = self._train_loop(train_iterator, pbar)
            
            train_history["epochs"].append(epoch_i)
            train_history["train"]["loss"].append(train_loss)
            train_history["train"]["error"].append(train_error)
            
            if is_validation_provided:
                val_loss, val_error = self._test_loop(val_iterator)
                
                train_history["validation"]["loss"].append(val_loss)
                train_history["validation"]["error"].append(val_error)
                
            
            if verbose >= 1 and is_validation_provided:
                print(f"Epoch {epoch_i:03d}/{epochs}: {train_loss = :.4f} ; {val_loss = :.4f} ; {train_error = :.4f} ; {val_error = :.4f}")
            
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
        
