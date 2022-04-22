# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import shutil
import torch
from sklearn import metrics as skmetrics
from sklearn.inspection import permutation_importance
from argparse import ArgumentParser
import shap

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning, get_hist, calc_pull
from utils.merge_pdfs import merge_pdfs
from model_B_classifier import DeepSetModel
from utils.data_handling import DataIteratorByEvents

# %%
# Constant variables
N_events = 50000 # how many events to use for calculating permutation importance

parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if "model_name" in args and isinstance(args.model_name, str):
    paths.update_B_classifier_name(args.model_name)
else:
    paths.update_B_classifier_name("B_classifier")
    
output_dir = paths.B_classifier_dir/"feature_importance"
if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

output_file = paths.B_classifier_dir/"feature_importance_B_classifier.pdf"

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
# Read in the model parameters
with open(paths.B_classifier_parameters_file, "r") as file:
    params = json.load(file)
    train_params = params["train_params"]
    model_params = params["model_params"]

label_key = params["label_key"]
feature_keys = params["feature_keys"]

feature_keys_in_X = ["event_id"] + feature_keys

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 input_file=paths.ss_classified_data_file,
                                 N_entries_max=10000000000)
print("Done reading input")

# Read in the train test split
with open(paths.B_classifier_train_test_split_file, "r") as file:
    ttsplit = json.load(file)
    
event_ids_train = ttsplit["train_ids"]
event_ids_test = ttsplit["test_ids"]

# %%
# Prepare the data
df_data.set_index(["event_id", "track_id"], drop=True, inplace=True)

# Get only the test data
temp_df = df_data.loc[event_ids_test[:20000],:]
X_test = temp_df.reset_index().loc[:,feature_keys_in_X].to_numpy()
y_test = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

# %%
# Read in the trained model
model = torch.load(paths.B_classifier_model_file).to(device)
    
    
# %%
# Prepare the Feature Importance DataFrame
df_fi = pd.DataFrame({"feature":feature_keys_in_X})
df_fi.set_index("feature", drop=True, inplace=True)

# %%
# Look at the weights of the first layer
df_fi.loc[feature_keys, "weights_abs_mean"] = np.mean(np.abs(next(model.parameters()).data.numpy()), axis=0)
df_fi.loc[feature_keys, "weights_abs_max"] = np.max(np.abs(next(model.parameters()).data.numpy()), axis=0)

# %%
torch.set_num_threads(20)

# %%
# Permutation Feature Importance through scikit-learn
# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
print("Calculate the permutation feature importance...")
perm_imp_metrics = ["accuracy", "precision", "recall", "balanced_accuracy", "roc_auc", "f1"]

n_repeats = 5
pbar = tqdm(total=(len(feature_keys_in_X)*n_repeats)+1, desc="Permutation Importance")

def ProgressCallback(y_true, y_pred):
    pbar.update(1)
    return 0

scoring_dict = {metric:metric for metric in perm_imp_metrics}
scoring_dict["progress_callback"] = skmetrics.make_scorer(ProgressCallback)

X_test = model._scale_X(X_test)

temp_scaler = model.scaler
model.scaler = None

perm_fi = permutation_importance(model, 
                                 X_test, 
                                 y_test,
                                 scoring=scoring_dict,
                                 n_repeats=n_repeats,
                                 n_jobs=1)

model.scaler = temp_scaler

for metric in perm_imp_metrics:
    df_fi.loc[feature_keys_in_X,f"perm_{metric}"] = perm_fi[metric]["importances_mean"]
    df_fi.loc[feature_keys_in_X,f"perm_{metric}_std"] = perm_fi[metric]["importances_std"]
    
print("Done calculating the permutation feature importance")

# %%
# remove the event_id as feature
df_fi.drop("event_id", inplace=True)

# %%
# Which importance metrics should be evaluated
importance_metrics = ["weights_abs_mean","weights_abs_max", "perm_balanced_accuracy", "perm_roc_auc", "perm_f1", "perm_accuracy", "perm_precision", "perm_recall"]

# %%
# Calculate a total score
# Scale all of them to 0-1 (min to max within one metric)
# then calculate the mean
# and the max
max_fi = np.max(df_fi[importance_metrics], axis=0)
min_fi = np.min(df_fi[importance_metrics], axis=0)
df_fi_normed = (df_fi[importance_metrics] - min_fi) / (max_fi - min_fi)

df_fi["combined_mean"] = np.mean(df_fi_normed, axis=1)
df_fi["combined_max"] = np.max(df_fi_normed, axis=1)

importance_metrics = ["combined_mean", "combined_max"] + importance_metrics

# %%
# Sort the features
df_fi.sort_values(by="combined_max", ascending=False, inplace=True)

# %%
# Plot the feature importances
fig, axs = plt.subplots(len(importance_metrics),1, 
                        figsize=(len(feature_keys)/1.5, len(importance_metrics)*5), 
                        sharex=True)

#fig.suptitle(f"Feature Importance")

for i, (ax, metric) in enumerate(zip(axs, importance_metrics)):
    ax.set_title(f"feature importance metric: {metric}")
    if f"{metric}_std" in df_fi.columns:
        err = df_fi[f"{metric}_std"]
    else:
        err = None
    ax.bar(df_fi.index, df_fi[metric], yerr=err, color=f"C{i}", alpha=0.8)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelbottom=True, labelrotation=60)

plt.tight_layout()
plt.savefig(output_dir/"00_selected_importances_horizontal.pdf")
plt.close()

# %%
# Plot the feature importances
fig, axs = plt.subplots(1,len(importance_metrics), 
                        figsize=(len(importance_metrics)*7, len(feature_keys)/1.5), 
                        sharey=False)

#fig.suptitle(f"Feature Importance")

for i, (ax, metric) in enumerate(zip(axs, importance_metrics)):
    ax.set_title(f"feature importance metric: {metric}")
    if f"{metric}_std" in df_fi.columns:
        err = df_fi[f"{metric}_std"]
    else:
        err = None
    ax.barh(df_fi.index, df_fi[metric], xerr=err, color=f"C{i}", alpha=0.8, zorder=3)

    ax.set_xlabel(metric)
    ax.tick_params(axis="y", left=True, labelleft=True)
    ax.tick_params(axis="x", bottom=True, top=True, labeltop=True)
    ax.grid(zorder=0)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir/"01_selected_importances_vertical.pdf")
plt.close()

# %%
# Merge all PDFs
merge_pdfs(output_dir,output_file)

# %%
