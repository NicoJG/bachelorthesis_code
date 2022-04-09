# %%
# Imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import json
import pickle

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data

# %%
# Constant variables
paths.ss_classifier_dir.mkdir(parents=True, exist_ok=True)

# Parameters of the model
params = {
    "test_size" : 0.4,
    "model_type" : "DecisionTreeClassifier",
    "model_params" : {
        "criterion" : "entropy",
        "max_depth" : 5,
        "class_weight" : "balanced",
        "min_samples_split" : 100
    },
    "train_params" : {
    }
    }
    # training parameters

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["label_ss_classifier","features_ss_classifier"])

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=feature_keys, 
                                 N_entries_max=10000000000)
print("Done reading input")


# %%
# Prepare the data
label_key = "Tr_is_SS"
feature_keys.remove(label_key)

X = df_data[feature_keys]
y = df_data[label_key].to_numpy()

# %%
# Split the data into train and test (with shuffling)
temp = train_test_split(X, y, 
                        test_size=params["test_size"], 
                        shuffle=True, 
                        stratify=y)
X_train, X_test, y_train, y_test = temp

print(f"Training Tracks: {len(y_train)}")
print(f"Test Tracks: {len(y_test)}")

params["n_tracks"] = len(y)
params["n_tracks_train"] = len(y_train)
params["n_tracks_test"] = len(y_test)


# %%
# Build the model
model = DecisionTreeClassifier(**params["model_params"])


# %%
# Train the model
model.fit(X_train, y_train)

# %%
# Save the indices of the train test split
train_idxs = X_train.index.to_list()
test_idxs = X_test.index.to_list()

with open(paths.ss_classifier_train_test_split_file, "w") as file:
    json.dump({"train_idxs":train_idxs,"test_idxs":test_idxs}, 
              fp=file, 
              indent=2)
    
# Save the parameters
with open(paths.ss_classifier_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
# Save the model
with open(paths.ss_classifier_model_file, "wb") as file:
    pickle.dump(model, file)
    
# %%
