# %%
# Imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import xgboost as xgb
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
    "n_estimators" : 2000,
    "max_depth" : 5,
    "learning_rate" : 0.1, # 0.3 is default
    "n_threads" : 50,
    "early_stopping_rounds" : 5000000,
    "objective" : "binary:logistic",
    "eval_metrics" : ["logloss", "error", "auc"]
}

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

# Save the indices of the train test split
train_idxs = X_train.index.to_list()
test_idxs = X_test.index.to_list()

with open(paths.ss_classifier_train_test_split_file, "w") as file:
    json.dump({"train_idxs":train_idxs,"test_idxs":test_idxs}, 
              fp=file, 
              indent=2)

# %%
# Build the BDT
params["scale_pos_weight"] = np.sum(y == 0)/np.sum(y == 1)

print(f"scale_pos_weight = {params['scale_pos_weight']}")

model = xgb.XGBClassifier(n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"], 
                    learning_rate=params["learning_rate"],
                    nthread=params["n_threads"],
                    objective=params["objective"],
                    scale_pos_weight=params["scale_pos_weight"],
                    use_label_encoder=False)

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
          eval_metric=params["eval_metrics"],
          early_stopping_rounds=params["early_stopping_rounds"],
          verbose=0,
          callbacks=[XGBProgressCallback(rounds=params["n_estimators"], 
                                         desc="BDT Train")]
          )

# %%
# Save the parameters
with open(paths.ss_classifier_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
# Save the model
with open(paths.ss_classifier_model_file, "wb") as file:
    pickle.dump(model, file)
    
# %%
