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
from model_B_classifier import DeepSetModel

# %%
# Constant variables
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if args.model_name is not None:
    paths.update_B_classifier_name(args.model_name)
else:
    paths.update_B_classifier_name("B_classifier")

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
event_ids = np.unique(df_data["event_id"].to_numpy())

df_data.set_index(["event_id", "track_id"], drop=True, inplace=True)

y = df_data.loc[(slice(None), 0), label_key].to_numpy()

# %%
# Split the data into train and test (with shuffling)
print("Split the data in training and test...")
event_ids_train, event_ids_test = train_test_split(event_ids, 
                                                   test_size=params["test_size"], 
                                                   shuffle=True, 
                                                   stratify=y)

temp_df = df_data.loc[event_ids_train,:]
X_train = temp_df.reset_index().loc[:,["event_id"]+feature_keys].to_numpy()
y_train = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

temp_df = df_data.loc[event_ids_test,:]
X_test = temp_df.reset_index().loc[:,["event_id"]+feature_keys].to_numpy()
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
# Build the model    
model = DeepSetModel(n_features=len(feature_keys), 
                     n_latent_features=len(feature_keys),
                     optimizer=torch.optim.Adam,
                     optimizer_kwargs={"lr":params["train_params"]["learning_rate"]},
                     loss=nn.BCELoss(),
                     scaler=StandardScaler())

model.to(device)

# %%
# Train the model
torch.set_num_threads(50)
model.fit(X_train, y_train, 
          X_val=X_test, y_val=y_test,
          epochs=params["train_params"]["epochs"],
          batch_size=params["train_params"]["batch_size"],
          verbose=1,
          device=device)

# %%
# Save everything to files

# Save the model
torch.save(model, paths.B_classifier_model_file)

# Save the parameters
with open(paths.B_classifier_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
# Save the train test split
with open(paths.B_classifier_train_test_split_file, "w") as file:
    json.dump({"train_ids":event_ids_train.tolist(),"test_ids":event_ids_test.tolist()}, 
              fp=file, 
              indent=2)

# %%
