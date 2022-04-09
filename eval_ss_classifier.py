# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import pickle
import sklearn.metrics as skmetrics

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.merge_pdfs import merge_pdfs
from utils.histograms import get_hist

# %%
# Constants
output_dir = paths.plots_dir/"eval_ss_classifier"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = paths.plots_dir/"eval_ss_classifier.pdf"

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["label_ss_classifier","features_ss_classifier"])

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=feature_keys, 
                                 N_entries_max=1000000000)
print("Done reading input")
    
# %%
# Prepare the data
label_key = "Tr_is_SS"
feature_keys.remove(label_key)

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
    
# Read in the model/training parameters
with open(paths.ss_classifier_parameters_file, "r") as file:
    params = json.load(file)
    train_params = params["train_params"]
    model_params = params["model_params"]

# %%
# Evaluate the training

# Plot the training history of multiple metrics
train_history = model.evals_result()
for i, metric in enumerate(train_params["eval_metric"]):
    iteration = list(range(len(train_history["validation_0"][metric])))
    plt.figure(figsize=(8, 6))
    plt.title(f"training performance ({metric})")
    plt.plot(iteration, train_history["validation_0"][metric], label="training data")
    plt.plot(iteration, train_history["validation_1"][metric], label="test data")
    plt.xlabel("iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(output_dir/f"00_train_performance_{i:02d}_{metric}.pdf")
    plt.show()

# %%
# Evaluate the model on test data

# get predictions
y_pred_proba_train = model.predict_proba(X_train)
y_pred_proba_test = model.predict_proba(X_test)

# %%
# Probability Distribution of both the train and the test data
plt.figure(figsize=(8,6))
plt.title("distribution of the prediction output of the BDT")

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

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.yscale("log")
plt.xlabel("BDT output")
plt.ylabel("density")
plt.legend()
plt.savefig(output_dir/"02_hist_output.pdf")
plt.show()

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

# %%
# ROC curve
auc = skmetrics.auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title(f"ROC curve, AUC={auc:.4f}")
plt.plot(fpr, tpr)
plt.xlabel("False positive rate (background)")
plt.ylabel("True positive rate (sameside)")
plt.savefig(output_dir/"03_roc.pdf")
plt.show()

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
plt.show()

# %%
# Plot various metrics for every cut
accuracy = (tp+tn)/(tp+tn+fp+fn)
#precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
balanced_accuracy = (recall+specificity)/2
signal_over_bkg = tp/(tn+fp+fn)

plt.figure(figsize=(8,6))
plt.title("Various metrics for every cut")
plt.plot(cut_linspace, accuracy, label="accuracy")
plt.plot(cut_linspace, balanced_accuracy, label="balanced accuracy")
#plt.plot(cut_linspace, precision, label="precision")
plt.plot(cut_linspace, recall, linestyle="dashed", label="recall/efficiency for SS")
plt.plot(cut_linspace, specificity, linestyle="dotted", label="specificity/efficiency for other")
plt.plot(cut_linspace, signal_over_bkg, linestyle="dashdot", label="TP/(TN+FP+FN)")
plt.xlabel("Cut")
plt.legend()
plt.savefig(output_dir/"05_metrics.pdf")
plt.show()

# %%
# Merge all evaluation plots
merge_pdfs(output_dir, output_file)

# %%
