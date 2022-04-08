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
    train_params = json.load(file)

# %%
# Evaluate the training

# Plot the training history of multiple metrics
train_history = model.evals_result()
for i, metric in enumerate(train_params["eval_metrics"]):
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
y_pred_proba = model.predict_proba(X_test)

y_pred_train_proba = model.predict_proba(X_train)

# %%
# Probability Distribution of the train data
plt.figure(figsize=(8,6))
plt.title("Distribution of the prediction output (train data)")
plt.hist(y_pred_train_proba[y_train==0][:,1], histtype="step", bins=200, range=(0.0, 1.0), density=True, label="other (ground truth)")
plt.hist(y_pred_train_proba[y_train==1][:,1], histtype="step", bins=200, range=(0.0, 1.0), density=True, label="SS (ground truth)")
plt.yscale("log")
plt.xlabel("Prediction Output of the SS classifier")
plt.ylabel("density (logarithmic)")
plt.legend()
plt.savefig(output_dir/"02_01_hist_proba_train.pdf")
plt.show()

# %%
# Probability Distribution of the test data
plt.figure(figsize=(8,6))
plt.title("Distribution of the prediction output (test data)")
plt.hist(y_pred_proba[y_test==0][:,1], histtype="step", bins=200, range=(0.0, 1.0), density=True, label="other (ground truth)")
plt.hist(y_pred_proba[y_test==1][:,1], histtype="step", bins=200, range=(0.0, 1.0), density=True, label="SS (ground truth)")
plt.yscale("log")
plt.xlabel("Prediction Output of the SS classifier")
plt.ylabel("density (logarithmic)")
plt.legend()
plt.savefig(output_dir/"02_02_hist_proba_test.pdf")
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
                                  y_test, y_pred_proba, 
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
