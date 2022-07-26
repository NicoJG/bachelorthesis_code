# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
import sklearn.metrics as skmetrics
from argparse import ArgumentParser

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
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

output_dir = paths.ss_classifier_eval_dir
output_dir.mkdir(exist_ok=True)

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
df_data = load_preprocessed_data(features=[label_key]+feature_keys, 
                                 N_entries_max=1000000000,
                                 n_threads=n_threads)
print("Done reading input")
    
# %%
# Prepare the data
X = df_data[feature_keys]
y = df_data[label_key].to_numpy()

# Read in the train test split
with open(paths.ss_classifier_train_test_split_file, "r") as file:
    ttsplit = json.load(file)
    
# Apply the train test split
X_train = X.loc[ttsplit["train_idxs"],:]
y_train = y[ttsplit["train_idxs"]]

X_test = X.loc[ttsplit["test_idxs"],:]
y_test = y[ttsplit["test_idxs"]]

# %%
# Read in the trained model
with open(paths.ss_classifier_model_file, "rb") as file:
    model = pickle.load(file)

# %%
# Evaluate the training

# Plot the training history of multiple metrics
train_history = model.evals_result()
for i, metric in enumerate(train_params["eval_metric"]):
    iteration = list(range(len(train_history["validation_0"][metric])))
    fig = plt.figure(figsize=(4,4))
    #plt.title(f"training performance ({metric})")
    plt.plot(iteration, train_history["validation_0"][metric], label="train data")
    plt.plot(iteration, train_history["validation_1"][metric], label="test data")
    plt.gca().set_box_aspect(1)
    plt.xlabel("training iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir/f"00_train_performance_{i:02d}_{metric}.pdf")
    #plt.show()
    plt.close()

# %%
# Evaluate the model on test data

# get predictions
y_pred_proba_train = model.predict_proba(X_train)
y_pred_proba_test = model.predict_proba(X_test)

# %%
# Probability Distribution of both the train and the test data
# binning
n_bins = 200
hist_range = (0.0, 1.0)
bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins+1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
bin_widths = np.diff(bin_edges)

y_pred_probas = [y_pred_proba_train[y_train==0][:,1], 
                 y_pred_proba_train[y_train==1][:,1],
                 y_pred_proba_test[y_test==0][:,1],
                 y_pred_proba_test[y_test==1][:,1]]
labels = ["other (train data)",
          "SS (train data)",
          "other (test data)", 
          "SS (test data)"]
colors = ["C0", "C1", "C0", "C1"]
plot_types = ["hist", "hist", "errorbar", "errorbar"]
alphas = [0.5, 0.5, 0.5, 0.5]

# Plot with log y-axis
plt.figure(figsize=(4,4))
#plt.title("distribution of the prediction output of the BDT")

plt.axvline(0.5, linestyle="dashed", color="grey")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.yscale("log")
plt.gca().set_box_aspect(1)
plt.xlabel("BDT output")
plt.ylabel("density")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(output_dir/"02_hist_output_logy.pdf")
plt.close()

# Plot with normal y-axis

plt.figure(figsize=(4,4))
#plt.title("distribution of the prediction output of the BDT")

plt.axvline(0.5, linestyle="dashed", color="grey")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.gca().set_box_aspect(1)
plt.xlabel("BDT output")
plt.ylabel("density")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(output_dir/"02_hist_output_normal.pdf")
plt.close()

# %%
# Analysis of different cuts
cut_linspace = np.linspace(0,1,1000)
    
def rates_for_cut(cut, y_true, y_pred_proba, pbar=None):
    y_pred = (y_pred_proba[:,1] >= cut).astype(int)
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
plt.figure(figsize=(4,4))
#plt.title(f"ROC curve, AUC=(test: {auc:.4f}, train: {auc_train:.4f})")
plt.plot([0,1],[0,1], "k--", label="no separation")
plt.plot(fpr_train, tpr_train, label="train data")
plt.plot(fpr, tpr, label="test data")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("False positive rate (other)")
plt.ylabel("True positive rate (SS)")
plt.legend(title=f"ROC AUC:\n  test:  {auc:.4f}\n  train: {auc_train:.4f})")
plt.tight_layout()
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
#precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
balanced_accuracy = (recall+specificity)/2
#f1_score = 2*tp/(2*tp+fp+fn)
#signal_over_bkg = tp/(tn+fp+fn)

recall_train = tp_train/(tp_train+fn_train)
specificity_train = tn_train/(tn_train+fp_train)
balanced_accuracy_train = (recall_train+specificity_train)/2

plt.figure(figsize=(8,6))
plt.title("Various metrics for every cut")
plt.plot(cut_linspace, accuracy, label="accuracy")
plt.plot(cut_linspace, balanced_accuracy, label="balanced accuracy")
#plt.plot(cut_linspace, precision, label="precision")
plt.plot(cut_linspace, recall, linestyle="dashed", label="recall/efficiency for SS/TPR")
plt.plot(cut_linspace, specificity, linestyle="dotted", label="specificity/efficiency for other/TNR")
#plt.plot(cut_linspace, f1_score, linestyle="dashdot", label="F1 score")

max_balanced_acc_idx = np.argmax(balanced_accuracy)
max_balanced_acc_cut = cut_linspace[max_balanced_acc_idx]
max_balanced_acc = balanced_accuracy[max_balanced_acc_idx]

plt.axvline(max_balanced_acc_cut, alpha=0.3, linestyle="dotted", label=f"max balanced acc : ({max_balanced_acc_cut:.4f}, {max_balanced_acc:.4f})")

plt.xlim(0.2,0.8)
plt.ylim(0.2,1.0)

plt.xlabel("Cut")
plt.legend()
plt.savefig(output_dir/"05_metrics.pdf")
plt.close()

# %%
# Merge all evaluation plots
merge_pdfs(output_dir, paths.ss_classifier_eval_plots_file)

# %%
eval_results = {
    "roc_auc_test" : float(auc),
    "max_balanced_accuracy_test" : float(balanced_accuracy[max_balanced_acc_idx]),
    "efficiency_SS_test" : float(recall[max_balanced_acc_idx]),
    "efficiency_other_test" : float(specificity[max_balanced_acc_idx]),
    "roc_auc_train" : float(auc_train),
    "max_balanced_accuracy_train" : float(balanced_accuracy_train[max_balanced_acc_idx]),
    "efficiency_SS_train" : float(recall_train[max_balanced_acc_idx]),
    "efficiency_other_train" : float(specificity_train[max_balanced_acc_idx])
}

with open(paths.ss_classifier_eval_data_file, "w") as file:
    json.dump(eval_results, file, indent=2)

# %%
