# %%
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import pickle
from argparse import ArgumentParser
import sklearn.metrics as skmetrics
import uproot

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.merge_pdfs import merge_pdfs
from utils.histograms import get_hist

# %%
# Constants
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", default="SS_classifier", help="name of the model directory")
parser.add_argument("-t", "--threads", dest="n_threads", default=5, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

model_name = args.model_name
paths.update_ss_classifier_name(model_name)
    
output_file = paths.ss_classified_data_file

output_feature_key = load_feature_keys(["output_ss_classifier"])[0]

# %%
# Read in the model parameters
with open(paths.ss_classifier_parameters_file, "r") as file:
    params = json.load(file)
    train_params = params["train_params"]
    model_params = params["model_params"]

label_key = params["label_key"]
feature_keys = params["feature_keys"]

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(input_file=paths.preprocessed_data_file, n_threads=n_threads)
print("Done reading input")

# %%
# Read in the trained model
with open(paths.ss_classifier_model_file, "rb") as file:
    model = pickle.load(file)

# %%
# Evaluate the training for comparison in the console output
print("training results on the test data:")
train_history = model.evals_result()
best_iter = model.best_iteration
for i, metric in enumerate(train_params["eval_metric"]):
    print(f"{metric}: {train_history['validation_1'][metric][best_iter]}")

print()

# %%
# Get predictions on the given data
print("apply the model to the given data")
y_pred_proba = model.predict_proba(df_data[feature_keys])[:,1]
df_data[output_feature_key] = y_pred_proba

# %%
# Evaluate the predictions if the label key is present
if label_key in df_data.columns:
    print(f"evaluate the predictions ({label_key} is present in the data)")
    y = df_data[label_key].values
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"ROC AUC: {skmetrics.roc_auc_score(y,y_pred_proba)}")
    print(f"balanced accuracy: {skmetrics.balanced_accuracy_score(y,y_pred)}")
    print(f"error rate: {np.sum(y != y_pred)/len(y)}")
    print(f"average precision: {skmetrics.average_precision_score(y, y_pred)}")
    
    print()
    
# %%
# Write the data to a new file
print("Writing output file...")

with uproot.recreate(output_file) as file:
    file["DecayTree"] = df_data
    
print("Done writing output")

# %%
