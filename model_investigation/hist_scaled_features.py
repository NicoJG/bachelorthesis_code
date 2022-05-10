# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json
import pickle
import shutil

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning, get_hist, calc_pull
from utils.merge_pdfs import merge_pdfs

# %%
# Constant variables

output_dir = paths.plots_dir / "scaled_features"

output_file = paths.plots_dir / "scaled_features.pdf"

# %%
# Read in the model parameters
with open(paths.B_classifier_parameters_file, "r") as file:
    params = json.load(file)
    train_params = params["train_params"]
    model_params = params["model_params"]

label_key = params["label_key"]
feature_keys = params["feature_keys"]

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
# Load the StandardScaler
with open(paths.B_classifier_scaler_file, "rb") as file:
    scaler = pickle.load(file)
    
# %%
# Prepare the data
df_data.set_index(["event_id", "track_id"], drop=True, inplace=True)

# Scale the data
df_data_scaled = df_data.copy()
df_data_scaled[feature_keys] = scaler.transform(df_data[feature_keys])
df_data_scaled[label_key] = df_data[label_key]

# %%
# Plot all scaled features by the label
def plot_feature(enumerated_fkey):
    i, fkey = enumerated_fkey
    
    fig = plt.figure(figsize=(5,7))
    
    ax = plt.subplot2grid(shape=(4,1), loc=(0,0), rowspan=3, colspan=1, fig=fig)
    pull_ax = plt.subplot2grid(shape=(4,1), loc=(3,0), rowspan=1, colspan=1, fig=fig)
    
    fig.suptitle(f"{fkey} by {label_key} after scaling")
    
    x_min = np.quantile(df_data_scaled[fkey], 0.01)
    x_max = np.quantile(df_data_scaled[fkey], 0.99)
    
    n_bins = 300
    
    if len(df_data_scaled[fkey].unique()) < 20:
        n_bins = 40
    
    bin_edges = np.linspace(x_min, x_max, n_bins+1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    
    x0_normed, sigma0_normed = get_hist(df_data_scaled.query(f"{label_key}==0")[fkey], bin_edges, normed=True)
    x1_normed, sigma1_normed = get_hist(df_data_scaled.query(f"{label_key}==1")[fkey], bin_edges, normed=True)
    
    pull = calc_pull(x0_normed, x1_normed, sigma0_normed, sigma1_normed)
    
    ax.hist(bin_centers, weights=x0_normed, bins=bin_edges, histtype="step", label="Bd", color="C0", alpha=0.8)
    ax.errorbar(bin_centers, x0_normed, yerr=sigma0_normed, fmt="none", color="C0",alpha=0.8)
    
    ax.hist(bin_centers, weights=x1_normed, bins=bin_edges, histtype="step", label="Bs", color="C1", alpha=0.8)
    ax.errorbar(bin_centers, x1_normed, yerr=sigma1_normed, fmt="none", color="C1",alpha=0.8)
    
    pull_ax.hist(bin_centers, weights=pull, bins=bin_edges, histtype="stepfilled",alpha=0.7)
    
    ax.set_ylabel("density")
    pull_ax.set_ylabel("pull")
    pull_ax.set_xlabel(f"{fkey} (standard scaled)")
    
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/f"{i:02d}_{fkey}.pdf")
    

if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

with Pool(processes=50) as p:
    pfunc = plot_feature
    iter = p.imap(pfunc, enumerate(feature_keys))
    pbar_iter = tqdm(iter, total=len(feature_keys), desc="Hist Features")
    # start processing, by evaluating the iterator:
    list(pbar_iter)
    
    
# %%
# Merge all plots
merge_pdfs(output_dir, output_file)

# %%
