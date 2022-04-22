# %%
# Imports
from lib2to3.pgen2.token import RARROW
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import pickle
import shutil
import torch
from torch import nn
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

# %%
# Constant variables
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if "model_name" in args and isinstance(args.model_name, str):
    paths.update_B_classifier_name(args.model_name)
else:
    paths.update_B_classifier_name("B_classifier")
    
output_dir = paths.B_classifier_eval_dir
if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

output_file = paths.B_classifier_eval_file

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

# Get only the test data
temp_df = df_data_scaled.loc[event_ids_train,:]
event_ids_train_by_track = temp_df.reset_index().loc[:,"event_id"].to_numpy()
X_train = temp_df.loc[:,feature_keys].to_numpy()
y_train = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

temp_df = df_data_scaled.loc[event_ids_test,:]
event_ids_test_by_track = temp_df.reset_index().loc[:,"event_id"].to_numpy()
X_test = temp_df.loc[:,feature_keys].to_numpy()
y_test = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

# %%
# Read in the trained model
model = torch.load(paths.B_classifier_model_file)

# %%
# Evaluate the training

# Read in the training history
with open(paths.B_classifier_training_history_file, "r") as file:
    train_history = json.load(file)

epochs = train_history["epochs"]

# Plot the training history of multiple metrics
for i, metric in enumerate(train_history["train"].keys()):
    plt.figure(figsize=(8, 6))
    plt.title(f"training performance ({metric})")
    plt.plot(epochs, train_history["train"][metric], label="training data")
    plt.plot(epochs, train_history["test"][metric], label="test data")
    plt.xlabel("iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(output_dir/f"00_train_performance_{i:02d}_{metric}.pdf")
    plt.close()

# %%
# Evaluate the model on test data

# get predictions
y_pred_proba_train = model(torch.from_numpy(X_train).float().to(device), torch.from_numpy(event_ids_train_by_track).int().to(device)).detach().numpy().flatten()
y_pred_proba_test = model(torch.from_numpy(X_test).float().to(device), torch.from_numpy(event_ids_test_by_track).int().to(device)).detach().numpy().flatten()

# %%
# Probability Distribution of both the train and the test data
# binning
n_bins = 200
hist_range = (0.0, 1.0)
bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins+1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
bin_widths = np.diff(bin_edges)

y_pred_probas = [y_pred_proba_train[y_train==0], 
                 y_pred_proba_train[y_train==1],
                 y_pred_proba_test[y_test==0],
                 y_pred_proba_test[y_test==1]]
labels = ["Bd (train data)",
          "Bs (train data)",
          "Bd (test data)", 
          "Bs (test data)"]
colors = ["C0", "C1", "C0", "C1"]
plot_types = ["hist", "hist", "errorbar", "errorbar"]
alphas = [0.5, 0.5, 0.5, 0.5]

# Plot with log y-axis
plt.figure(figsize=(8,6))
plt.title("distribution of the prediction output of the DeepSet")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.yscale("log")
plt.xlabel("DeepSet output")
plt.ylabel("density")
plt.legend()
plt.savefig(output_dir/"02_hist_output_logy.pdf")
plt.close()

# Plot with normal y-axis
plt.figure(figsize=(8,6))
plt.title("distribution of the prediction output of the DeepSet")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.xlabel("DeepSet output")
plt.ylabel("density")
plt.legend()
plt.savefig(output_dir/"02_hist_output_normal.pdf")
plt.close()

# %%
# Analysis of different cuts
cut_linspace = np.linspace(0,1,1000)
    
def rates_for_cut(cut, y_true, y_pred_proba, pbar=None):
    y_pred = (y_pred_proba >= cut).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    if isinstance(pbar, tqdm):
        pbar.update(1)
    return tp,fp,fn,tn

tp,fp,fn,tn = np.apply_along_axis(rates_for_cut, 
                                  1,
                                  cut_linspace[:,np.newaxis], 
                                  y_test, y_pred_proba_test, 
                                  tqdm(total=len(cut_linspace), 
                                       desc="Calc tp,fp,fn,tn")).T

# calculate the rates
tpr = tp/(tp+fn)
fnr = fn/(tp+fn)
tnr = tn/(tn+fp)
fpr = fp/(tn+fp)

# for training data
tp_train,fp_train,fn_train,tn_train = np.apply_along_axis(rates_for_cut, 
                                  1,
                                  cut_linspace[:,np.newaxis], 
                                  y_train, y_pred_proba_train, 
                                  tqdm(total=len(cut_linspace), 
                                       desc="Calc tp,fp,fn,tn")).T

# calculate the rates
tpr_train = tp_train/(tp_train+fn_train)
fpr_train = fp_train/(tn_train+fp_train)

# %%
# ROC curve
auc = skmetrics.auc(fpr, tpr)
auc_train = skmetrics.auc(fpr_train, tpr_train)
plt.figure(figsize=(8, 6))
plt.title(f"ROC curve, AUC=(test: {auc:.4f}, train: {auc_train:.4f})")
plt.plot(fpr, tpr, label="test data")
plt.plot(fpr_train, tpr_train, alpha=0.5, label="train data")
plt.xlabel("False positive rate (background)")
plt.ylabel("True positive rate (sameside)")
plt.legend()
plt.savefig(output_dir/"03_roc.pdf")
plt.close()

# %%
# Plot the rates for every cut

plt.figure(figsize=(8,6))
plt.title("Prediction rates for every cut")
plt.plot(cut_linspace, tpr, label="tpr (True Positive Rate)", linestyle="dashed")
plt.plot(cut_linspace, fnr, label="fnr (False Negative Rate)", linestyle="dashed")
plt.plot(cut_linspace, tnr, label="tnr (True Negative Rate)")
plt.plot(cut_linspace, fpr, label="fpr (False Positive Rate)")
plt.xlabel("Cut")
#plt.yscale("log")
plt.legend()
plt.savefig(output_dir/"04_pred_rates.pdf")
plt.close()

# %%
# Plot various metrics for every cut
accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
balanced_accuracy = (recall+specificity)/2
#f1_score = 2*tp/(2*tp+fp+fn)
#signal_over_bkg = tp/(tn+fp+fn)

plt.figure(figsize=(8,6))
plt.title("Various metrics for every cut")
plt.plot(cut_linspace, accuracy, label="accuracy")
plt.plot(cut_linspace, balanced_accuracy, label="balanced accuracy")
plt.plot(cut_linspace, precision, label="precision")
plt.plot(cut_linspace, recall, linestyle="dashed", label="recall/efficiency for Bs/TPR")
plt.plot(cut_linspace, specificity, linestyle="dotted", label="specificity/efficiency for Bd/TNR")
#plt.plot(cut_linspace, f1_score, linestyle="dashdot", label="F1 score")

max_balanced_acc_idx = np.argmax(balanced_accuracy)
max_balanced_acc_cut = cut_linspace[max_balanced_acc_idx]
max_balanced_acc = balanced_accuracy[max_balanced_acc_idx]

plt.axvline(max_balanced_acc_cut, alpha=0.3, linestyle="dotted", label=f"max balanced acc : ({max_balanced_acc_cut:.4f}, {max_balanced_acc:.4f})")

plt.xlabel("Cut")
plt.legend()
plt.savefig(output_dir/"05_metrics.pdf")
plt.close()

# %%
# Merge all evaluation plots
merge_pdfs(output_dir, output_file)

# %%
