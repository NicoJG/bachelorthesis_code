# %%
# Imports
from turtle import forward
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
import pickle
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data

# %%
# Constant variables
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if "model_name" in args and isinstance(args.model_name, str):
    paths.update_ss_classifier_name(args.model_name)
else:
    paths.update_ss_classifier_name("SS_classifier_NN")

# assert not paths.ss_classifier_dir.is_dir(), f"The model '{paths.ss_classifier_dir}' already exists! To overwrite it please (re-)move this directory or choose another model name with the flag '--model_name'."
paths.ss_classifier_dir.mkdir(parents=True, exist_ok=True)

# %%
# Parameters of the model
params = {
    "model_type" : "NN_PyTorch",
    "test_size" : 0.4,
    "model_params" : {
        
    },
    "train_params" : {
        "batch_size" : 10000,
        "learning_rate" : 0.001,
        "epochs" : 100,
        "loss" : "binary_crossentropy",
        "optimizer" : "Adam"
    }
    }


# %%
# Read in the feature keys
feature_keys = load_feature_keys(["features_ss_classifier"])
label_key = load_feature_keys(["label_ss_classifier"])[0]

params["label_key"] = label_key
params["feature_keys"] = feature_keys

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 N_entries_max=1000000)
print("Done reading input")


# %%
# Prepare the data
X = df_data[feature_keys].values
y = df_data[label_key].values

X_scaled = StandardScaler().fit_transform(X)

# %%
# Split the data into train and test (with shuffling)
df_idxs = df_data.index.to_numpy()
np_idxs = np.arange(X.shape[0])

temp = train_test_split(df_idxs, np_idxs, 
                        test_size=params["test_size"], 
                        shuffle=True, 
                        stratify=y)
train_df_idxs, test_df_idxs, train_np_idxs, test_np_idxs = temp

X_train = X_scaled[train_np_idxs]
y_train = y[train_np_idxs]

X_test = X_scaled[test_np_idxs]
y_test = y[test_np_idxs]

print(f"Training Tracks: {len(y_train)}")
print(f"Test Tracks: {len(y_test)}")

params["n_tracks"] = len(y)
params["n_tracks_train"] = len(y_train)
params["n_tracks_test"] = len(y_test)

# %%
# Prepare the dataloaders
tensor_X_train = torch.from_numpy(X_train).float()
tensor_y_train = torch.from_numpy(y_train)
train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
train_dataloader = DataLoader(train_dataset, batch_size=params["train_params"]["batch_size"])

tensor_X_test = torch.from_numpy(X_test).float()
tensor_y_test = torch.from_numpy(y_test)
test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
test_dataloader = DataLoader(test_dataset, batch_size=params["train_params"]["batch_size"])

# %%
# Build the NN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(feature_keys), 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(1)
        )
        
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

model = NeuralNetwork().to(device)
print(model)

# %%
# Define the training functions
def train_loop(dataloader, model, loss_fn, optimizer, pbar=None):
    train_loss = 0
    error_count = 0
    
    for X, y in dataloader:
        # Compute prediction and loss
        y_pred = model(X)
        loss = loss_fn(y_pred, nn.functional.one_hot(y, num_classes=2).float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        error_count += (y_pred.argmax(1) != y).type(torch.float).sum().item()
        if pbar is not None:
            pbar.update()
        
    train_loss /= len(dataloader)
    train_error = error_count / len(dataloader.dataset)
    
    return train_loss, train_error


def test_loop(dataloader, model, loss_fn):
    test_loss = 0
    error_count = 0
    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            test_loss += loss_fn(y_pred, nn.functional.one_hot(y, num_classes=2).float()).item()
            error_count += (y_pred.argmax(1) != y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    test_error = error_count / len(dataloader.dataset)

    return test_loss, test_error

# %%
# Train the model
torch.set_num_threads(50)
loss_fn = nn.CrossEntropyLoss()
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
    train_loss, train_error = train_loop(train_dataloader, model, loss_fn, optimizer, pbar)
    test_loss, test_error = test_loop(test_dataloader, model, loss_fn)
    
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
y_pred = model(tensor_X_test)


# %%
# Save the indices of the train test split
with open(paths.ss_classifier_train_test_split_file, "w") as file:
    json.dump({"train_idxs":train_df_idxs,"test_idxs":test_df_idxs}, 
              fp=file, 
              indent=2)
    
# Save the parameters
with open(paths.ss_classifier_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
with open(paths.ss_classifier_dir/"train_history.json", "w") as file:
    json.dump(train_history, file, indent=2)
    
# Save the model
torch.save(model, paths.ss_classifier_model_file)
    
# %%
