# %%
# Imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import json
from argparse import ArgumentParser
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import pickle

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_preprocessed_data

# %%
# Constant variables
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if args.model_name is not None:
    paths.update_B_classifier_name(args.model_name)

assert not paths.B_classifier_dir.is_dir(), f"The model '{paths.B_classifier_dir}' already exists! To overwrite it please (re-)move this directory or choose another model name with the flag '--model_name'."
paths.B_classifier_dir.mkdir(parents=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
# Parameters of the model
params = {
    "test_size" : 0.4,
    "model_params" : {
        
    },
    "train_params" : {
        "batch_size" : 10000,
        "learning_rate" : 0.001,
        "epochs" : 10
    }

}

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["features_B_classifier"])
label_key = load_feature_keys(["label_B_classifier"])[0]

params["label_key"] = label_key
params["feature_keys"] = feature_keys

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 input_file=paths.ss_classified_data_file,
                                 N_entries_max=10000000000)
print("Done reading input")


# %%
# Prepare the data
event_ids_by_track = df_data["event_id"].to_numpy()
event_ids = np.unique(event_ids_by_track)

df_data.set_index(["event_id", "track_id"], drop=True, inplace=True)

y = df_data.loc[(slice(None), 0), label_key].to_numpy()

# %%
# Scale the features with sklearn.preprocessing.StandardScaler
scaler = StandardScaler()
df_data_scaled = df_data.copy()
df_data_scaled[feature_keys] = scaler.fit_transform(df_data[feature_keys])
df_data_scaled[label_key] = df_data[label_key]

# %%
# Split the data into train and test (with shuffling)
print("Split the data in training and test...")
event_ids_train, event_ids_test = train_test_split(event_ids, 
                                                   test_size=params["test_size"], 
                                                   shuffle=True, 
                                                   stratify=y)

temp_df = df_data_scaled.loc[event_ids_train,:]
event_ids_train_by_track = temp_df.reset_index().loc[:,"event_id"].to_numpy()
X_train = temp_df.loc[:,feature_keys].to_numpy()
y_train = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

temp_df = df_data_scaled.loc[event_ids_test,:]
event_ids_test_by_track = temp_df.reset_index().loc[:,"event_id"].to_numpy()
X_test = temp_df.loc[:,feature_keys].to_numpy()
y_test = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

# print(event_ids_train.shape, event_ids_train_by_track.shape, X_train.shape, y_train.shape)
# print(event_ids_test.shape, event_ids_test_by_track.shape, X_test.shape, y_test.shape)

print(f"All Events: {len(event_ids)}")
print(f"Training Events: {len(event_ids_train)}")
print(f"Test Events: {len(event_ids_test)}")

params["n_events"] = len(event_ids)
params["n_events_train"] = len(event_ids_train)
params["n_events_test"] = len(event_ids_test)

# %%
# Create PyTorch Tensors from the Numpy arrays


X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
event_ids_train_by_track = torch.from_numpy(event_ids_train_by_track).int().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
event_ids_test_by_track = torch.from_numpy(event_ids_test_by_track).int().to(device)


# %%
# Define the model
# similar to https://gitlab.cern.ch/nnolte/DeepSetTagging/-/blob/cf286579b7db4ff21341eb4b06fc8f726168a8c9/model.py
class DeepSetModel(nn.Module):
    def __init__(self, n_features, n_latent_features):
        global SumLayer
        super(DeepSetModel, self).__init__()
        
        self.n_features = n_features
        self.n_latent_features = n_latent_features
        
        # Deep Set structure:
        self.phi_stack = nn.Sequential(
            nn.Linear(n_features, n_features*2),
            nn.ReLU(),
            nn.Linear(n_features*2, n_latent_features),
            nn.ReLU()
        )

        def SumLayer(x, ids):
            # get an event_index_by_track array
            idxs = torch.zeros_like(ids)
            idxs[1:] = torch.cumsum(ids[1:] != ids[:-1], 0)
            
            # sum up the latent features of all tracks per event
            temp = torch.zeros(idxs[-1]+1, n_latent_features)
            return temp.index_add(0, idxs, x)
            
        self.sum_layer = SumLayer
        
        self.rho_stack = nn.Sequential(
            nn.Linear(n_latent_features, n_latent_features*2),
            nn.ReLU(),
            nn.Linear(n_latent_features*2, n_latent_features),
            nn.ReLU(),
            nn.Linear(n_latent_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, event_ids):
        # x must have shape (tracks, features)
        # event_ids must have shape (tracks,)
        x = self.phi_stack(x)
        x = self.sum_layer(x, event_ids)
        x = self.rho_stack(x)
        return x
        
model = DeepSetModel(len(feature_keys), len(feature_keys)).to(device)

# %%
# Define a dataloader
# based on https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class DeepSetDataLoader:
    def __init__(self, X, y, event_ids_by_track, batch_size):
        self.X = X
        self.y = y.unsqueeze(1)
        
        self.event_ids_by_track = event_ids_by_track
        self.event_ids, self.event_first_idxs = np.unique(event_ids_by_track.numpy(), return_index=True)
        # numpy unique sorts the values, so we have to "unsort" them
        unsort_mask = np.argsort(self.event_first_idxs)
        self.event_ids = torch.from_numpy(self.event_ids[unsort_mask])
        self.event_first_idxs = torch.from_numpy(self.event_first_idxs[unsort_mask])
        
        self.batch_size = batch_size
        
        assert X.shape[0] == self.event_ids_by_track.shape[0], "X and event_ids_by_track must have the same length in the first dimension!"
        assert y.shape[0] == self.event_ids.shape[0], "y must have the same length as unique values in event_ids_by_track!"
        
        self.n_events = len(self.event_ids)
        self.n_tracks = len(self.event_ids_by_track)
        self.n_batches = int(np.ceil(self.n_events / self.batch_size))
        
    def __iter__(self):
        self.current_event_idx = 0
        return self
        
    def __next__(self):
        if self.current_event_idx >= self.n_events:
            raise StopIteration
        
        batch_start_event_idx = self.current_event_idx
        batch_stop_event_idx = batch_start_event_idx + self.batch_size # index of the first event that is not in the batch
        
        batch_start_track_idx = self.event_first_idxs[batch_start_event_idx]
        
        if batch_stop_event_idx >= self.n_events:
            # last batch in the dataset
            batch_stop_event_idx = self.n_events
            batch_stop_track_idx = self.n_tracks
        else:
            batch_stop_track_idx = self.event_first_idxs[batch_stop_event_idx]
            
        batch_event_slice = slice(batch_start_event_idx, batch_stop_event_idx)
        batch_track_slice = slice(batch_start_track_idx, batch_stop_track_idx)
        
        self.current_event_idx = batch_stop_event_idx
        
        return (self.X[batch_track_slice],
                self.y[batch_event_slice],
                self.event_ids_by_track[batch_track_slice],
                self.event_ids[batch_event_slice]) 
        
    def __len__(self):
        return self.n_batches
        
train_dataloader = DeepSetDataLoader(X_train, y_train, event_ids_train_by_track, batch_size=params["train_params"]["batch_size"])
test_dataloader = DeepSetDataLoader(X_test, y_test, event_ids_test_by_track, batch_size=params["train_params"]["batch_size"])
            
# %%
# Define the training loop functions
def train_one_epoch(dataloader, model, loss_fn, optimizer, pbar=None):
    loss_sum = 0
    error_count = 0
    
    model.train()
    
    for i, (X, y, event_ids_by_track, event_ids) in enumerate(dataloader):
        # Compute prediction and loss
        y_pred = model(X, event_ids_by_track)
        loss = loss_fn(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        batch_error = ((y_pred > 0.5) != y).sum().item()
        
        print(f"Batch {i:03d}/{dataloader.n_batches}: loss={batch_loss} error={batch_error/len(y)}")

        loss_sum += batch_loss
        error_count += batch_error
        if pbar is not None:
            pbar.update()
        
    loss = loss_sum / dataloader.n_batches
    error = error_count / dataloader.n_events
    
    return loss, error

def test_one_epoch(dataloader, model, loss_fn):
    loss_sum = 0
    error_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for X, y, event_ids_by_track, event_ids in dataloader:
            y_pred = model(X, event_ids_by_track)
            loss_sum += loss_fn(y_pred, y).item()
            error_count += ((y_pred > 0.5) != y).sum().item()

    loss = loss_sum / dataloader.n_batches
    error = error_count / dataloader.n_events

    return loss, error

    

# %%
# Train the model
torch.set_num_threads(50)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["train_params"]["learning_rate"])

train_history = {"epochs" : [],
                 "train": {"loss":[],"error":[]},
                 "test": {"loss":[],"error":[]}}


for epoch_i in tqdm(range(params["train_params"]["epochs"]), desc="Train epochs"):
    if epoch_i == 0:
        pbar = tqdm(total=len(train_dataloader), desc="Batches")
    else:
        pbar.refresh()
        pbar.reset()
    train_loss, train_error = train_one_epoch(train_dataloader, model, loss_fn, optimizer, pbar)
    test_loss, test_error = test_one_epoch(test_dataloader, model, loss_fn)
    
    train_history["epochs"].append(epoch_i)
    train_history["train"]["loss"].append(train_loss)
    train_history["train"]["error"].append(train_error)
    train_history["test"]["loss"].append(test_loss)
    train_history["test"]["error"].append(test_error)

print("Done!")

# %%
# Show the training history
import matplotlib.pyplot as plt
plt.title("Loss during training")
plt.plot(train_history["epochs"], train_history["train"]["loss"], label="train")
plt.plot(train_history["epochs"], train_history["test"]["loss"], label="test")
plt.legend()
plt.tight_layout()
plt.show()

plt.title("Error rate during training")
plt.plot(train_history["epochs"], train_history["train"]["error"], label="train")
plt.plot(train_history["epochs"], train_history["test"]["error"], label="test")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Save everything to files

# Save the model
torch.save(model, paths.B_classifier_model_file)

# Save the parameters
with open(paths.B_classifier_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
# Save the train test split
with open(paths.ss_classifier_train_test_split_file, "w") as file:
    json.dump({"train_ids":event_ids_train.tolist(),"test_ids":event_ids_test.tolist()}, 
              fp=file, 
              indent=2)
    
# Save the training history
with open(paths.B_classifier_training_history_file, "w") as file:
    json.dump(train_history, file, indent=2)
    
# Save the sklearn.preprocessing.StandardScaler
with open(paths.B_classifier_scaler_file, "wb") as file:
    pickle.dump(scaler, file)

# %%
