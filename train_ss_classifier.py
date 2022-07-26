# %%
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json
import pickle
from argparse import ArgumentParser

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data

# %%
# Constant variables
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", default="SS_classifier", help="name of the model directory")
parser.add_argument("-g", "--gpu", dest="train_on_gpu", action="store_true")
parser.add_argument("-t", "--threads", dest="n_threads", default=5, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

model_name = args.model_name
paths.update_ss_classifier_name(model_name)

assert not paths.ss_classifier_model_file.is_file(), f"The model '{paths.ss_classifier_model_file}' already exists! To overwrite it please (re-)move this directory or choose another model name with the flag '--model_name'."

# Parameters of the model
params = {
    "test_size" : 0.4,
    "model_params" : {
        "n_estimators" : 2000,
        "max_depth" : 4,
        "learning_rate" : 0.1, # 0.3 is default
        #"max_delta_step" : 0,
        #"reg_lambda" : 1.0, # L2 regularization
        #"subsample" : 1.0,
        "scale_pos_weight" : "TO BE SET", # sum(negative instances) / sum(positive instances)
        "objective" : "binary:logistic",
        "nthread" : n_threads,
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
if args.train_on_gpu:
    params["tree_method"] = "gpu_hist"
    print("Training on GPUs")

# %%
# Read in the feature keys
feature_keys = load_feature_keys([f"features_{model_name}"], file_path=paths.features_SS_classifier_file)
label_key = load_feature_keys(["label_ss_classifier"])[0]
params["label_key"] = label_key
params["feature_keys"] = feature_keys

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 N_entries_max=10000000000,
                                 n_threads=n_threads)
print("Done reading input")


# %%
# Prepare the data

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

paths.ss_classifier_dir.mkdir(parents=True, exist_ok=True)

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
