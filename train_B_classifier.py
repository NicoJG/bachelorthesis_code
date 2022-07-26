# %%
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
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
parser.add_argument("-n", "--model_name", dest="model_name", default="B_classifier", help="name of the model directory")
parser.add_argument("-g", "--gpu", dest="train_on_gpu", action="store_true", help="Flag to train on a GPU/CUDA")
parser.add_argument("-t", "--threads", dest="n_threads", default=5, type=int, help="Number of threads to use.")
parser.add_argument("-l", "--log_mode", dest="log_mode", help="if the output is written to a file or to a console", action="store_true")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

log_mode = args.log_mode

if log_mode:
    print("Printing is in log mode.")

model_name = args.model_name
paths.update_B_classifier_name(model_name)

assert not paths.B_classifier_model_file.is_file(), f"The model '{paths.B_classifier_model_file}' already exists! To overwrite it please (re-)move this directory or choose another model name with the flag '--model_name'."

n_threads = args.n_threads
assert n_threads > 0

# Get cpu or gpu device for training.
if torch.cuda.is_available():
    print("CUDA is available.")
device = "cuda" if args.train_on_gpu and torch.cuda.is_available() else "cpu"
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
        "epochs" : 1000,
        "early_stopping_epochs" : 50
    }

}

# %%
# Read in the feature keys
feature_keys = load_feature_keys([f"features_{model_name}"], file_path=paths.features_B_classifier_file)
label_key = load_feature_keys(["label_B_classifier"])[0]

params["label_key"] = label_key
params["feature_keys"] = feature_keys

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 input_file=paths.ss_classified_data_file,
                                 n_threads=n_threads)
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
                     optimizer=torch.optim.Adam,
                     optimizer_kwargs={"lr":params["train_params"]["learning_rate"]},
                     loss=nn.BCELoss(),
                     scaler=StandardScaler())

model.to(device)

# %%
# Train the model
torch.set_num_threads(n_threads)
model.fit(X_train, y_train, 
          X_val=X_test, y_val=y_test,
          epochs=params["train_params"]["epochs"],
          batch_size=params["train_params"]["batch_size"],
          device=device,
          show_epoch_progress=True,
          show_epoch_eval=log_mode,
          show_batch_progress=(not log_mode),
          show_batch_eval=False,
          early_stopping_epochs=params["train_params"]["early_stopping_epochs"])

model.to("cpu")

# %%
# Save everything to files
paths.B_classifier_dir.mkdir(parents=True, exist_ok=True)

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
