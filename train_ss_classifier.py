# %%
# Imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json
import pickle
from datetime import datetime

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data

# %%
# Constant variables
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
paths.update_ss_classifier_dir(f"SS_classifier_{datetime_str}")

paths.ss_classifier_dir.mkdir(parents=True, exist_ok=True)

# Parameters of the model
params = {
    "test_size" : 0.4,
    "model_params" : {
        "n_estimators" : 100,
        "max_depth" : 4,
        "learning_rate" : 0.1, # 0.3 is default
        #"max_delta_step" : 0,
        #"reg_lambda" : 1.0, # L2 regularization
        #"subsample" : 1.0,
        "scale_pos_weight" : "TO BE SET", # sum(negative instances) / sum(positive instances)
        "objective" : "binary:logistic",
        "nthreads" : 50,
        "tree_method" : "hist",
        #"num_parallel_tree" : 1
    },
    "train_params" : {
        #"early_stopping_rounds" : 50,
        "eval_metric" : ["logloss", "error", "auc", "aucpr", "map"],
        "verbose" : 0,
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

params["label_key"] = label_key
params["feature_keys"] = feature_keys

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
# Build the BDT
scale_pos_weight = np.sum(y == 0)/np.sum(y == 1)
params["model_params"]["scale_pos_weight"] = scale_pos_weight

print(f"scale_pos_weight = {scale_pos_weight}")

model = xgb.XGBClassifier(**params["model_params"], use_label_encoder=False)

# %%
# Callback for a progress bar of the training
class XGBProgressCallback(xgb.callback.TrainingCallback):
    """Show a progressbar using TQDM while training"""
    def __init__(self, rounds=None, desc=None):
        self.pbar = tqdm(total=rounds, desc=desc)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

# %%
# Train the BDT
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          **params["train_params"],
          callbacks=[XGBProgressCallback(rounds=params["model_params"]["n_estimators"], desc="BDT Train")]
          )

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
