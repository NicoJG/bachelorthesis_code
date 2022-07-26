# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil
import torch
from sklearn import metrics as skmetrics
from argparse import ArgumentParser

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
parser.add_argument("-g", "--gpu", dest="eval_on_gpu", action="store_true")
parser.add_argument("-t", "--threads", dest="n_threads", default=5, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if args.model_name is not None:
    paths.update_B_classifier_name(args.model_name)
else:
    paths.update_B_classifier_name("B_classifier")
    
output_dir = paths.B_classifier_eval_dir
if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

n_threads = args.n_threads

assert n_threads > 0

# Get cpu or gpu device for training.
device = "cuda" if args.eval_on_gpu and torch.cuda.is_available() else "cpu"
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
                                 n_threads=n_threads)
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
temp_df = df_data.loc[event_ids_train,:]
X_train = temp_df.reset_index().loc[:,["event_id"]+feature_keys].to_numpy()
y_train = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

temp_df = df_data.loc[event_ids_test,:]
X_test = temp_df.reset_index().loc[:,["event_id"]+feature_keys].to_numpy()
y_test = temp_df.loc[(slice(None),0),label_key].to_numpy()
del temp_df

# %%
# Read in the trained model
model = torch.load(paths.B_classifier_model_file, map_location=device)

# %%
# Evaluate the training

# Read in the training history
epochs = model.train_history["epochs"]

# Plot the training history of multiple metrics
for i, metric in enumerate(model.train_history["eval_metrics"]):
    plt.figure(figsize=(4,4))
    #plt.title(f"training performance ({metric})")
    
    if "best_epoch" in model.train_history.keys():
        plt.axvline(model.train_history["best_epoch"], color="black", alpha=0.5, linestyle="dashed", label=f"best epoch: {model.train_history['best_epoch']}")
        
    #if "train_wrong" in model.train_history:
    #    plt.plot(epochs, model.train_history["train_wrong"][metric], label="training data (with dropout)")
        
    plt.plot(epochs, model.train_history["train"][metric], label="train data")
    
    if "validation" in model.train_history.keys():
        plt.plot(epochs, model.train_history["validation"][metric], label="test data")
           
    plt.gca().set_box_aspect(1)
    plt.xlabel("training iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir/f"00_train_performance_{i:02d}_{metric}.pdf")
    plt.close()

# %%
# Evaluate the model on test data
torch.set_num_threads(n_threads)

# get predictions
y_pred_proba_train = model.decision_function(X_train)
y_pred_proba_test = model.decision_function(X_test)



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
plt.figure(figsize=(4,4))
#plt.title(f"ROC curve, AUC=(test: {auc:.4f}, train: {auc_train:.4f})")
plt.plot([0,1],[0,1], "k--", label="no separation")
plt.plot(fpr_train, tpr_train, label="train data")
plt.plot(fpr, tpr, label="test data")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r"False positive rate ($B_d$)")
plt.ylabel(r"True positive rate ($B_s$)")
plt.legend(title=f"ROC AUC:\n  test:  {auc:.4f}\n  train: {auc_train:.4f})")
plt.tight_layout()
plt.savefig(output_dir/"03_roc.pdf")
plt.close()

# %%
# Plot the confusion matrix with a few metrics
y_pred_train = (y_pred_proba_train > 0.5).astype(int)
y_pred_test = (y_pred_proba_test > 0.5).astype(int)

conf_mat = skmetrics.confusion_matrix(y_test, y_pred_test)

tn = conf_mat[0,0]
fp = conf_mat[0,1]
fn = conf_mat[1,0]
tp = conf_mat[1,1]

accuracy_test = (tp+tn)/(tp+fn+tn+fp)
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)

accuracy_train = skmetrics.accuracy_score(y_train, y_pred_train)

fig, ax = plt.subplots(figsize=(8,6))
conf_mat_display = skmetrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=["Bd", "Bs"], normalize="true", ax=ax, cmap="Blues", values_format=".4f")
plt.title(f"""Confusion Matrix (normed on the true labels)
accuracy(test):  {accuracy_test:.4f}
accuracy(train): {accuracy_train:.4f}""")
plt.tight_layout()
plt.savefig(output_dir/"04_confusion_matrix.pdf")
plt.close()


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
labels = [r"$B_d$ (train data)",
          r"$B_s$ (train data)",
          r"$B_d$ (test data)", 
          r"$B_s$ (test data)"]
colors = ["C0", "C1", "C0", "C1"]
plot_types = ["hist", "hist", "errorbar", "errorbar"]
alphas = [0.5, 0.5, 0.5, 0.5]

# Plot with log y-axis
plt.figure(figsize=(8,6))
plt.title(f"distribution of the prediction output of the DeepSet\nwith cut at 0.5: test accuracy = {accuracy_test:.3f} , train accuracy = {accuracy_train:.3f}\nefficiency Bd={tnr:.3f}, efficiency Bs={tpr:.3f}")

plt.axvline(0.5, linestyle="dashed", color="grey")

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
plt.tight_layout()
plt.savefig(output_dir/"02_hist_output_logy_old.pdf")
plt.close()

# Plot with log y-axis
plt.figure(figsize=(4,4))
#plt.title(f"distribution of the prediction output of the DeepSet\nwith cut at 0.5: test accuracy = {accuracy_test:.3f} , train accuracy = {accuracy_train:.3f}\nefficiency Bd={tnr:.3f}, efficiency Bs={tpr:.3f}")

plt.axvline(0.5, linestyle="dashed", color="grey")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.yscale("log")
plt.gca().set_box_aspect(1)
plt.xlabel("DeepSet output")
plt.ylabel("density")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(output_dir/"02_hist_output_logy.pdf")
plt.close()

# Plot with normal y-axis
plt.figure(figsize=(4,4))
#plt.title(f"distribution of the prediction output of the DeepSet\nwith cut at 0.5: test accuracy = {accuracy_test:.3f} , train accuracy = {accuracy_train:.3f}\nefficiency Bd={tnr:.3f}, efficiency Bs={tpr:.3f}")

plt.axvline(0.5, linestyle="dashed", color="grey")

for y_pred_proba, l, c, pt, a in zip(y_pred_probas, labels, colors, plot_types, alphas):
    x, sigma = get_hist(y_pred_proba, bin_edges, normed=True)
    if pt == "hist":
        plt.hist(bin_centers, weights=x, bins=bin_edges, histtype="stepfilled", color=c, alpha=a, label=l)
    elif pt == "errorbar":
        plt.errorbar(bin_centers, x, yerr=sigma, xerr=bin_widths/2, ecolor=c, label=l, fmt="none", elinewidth=1.0)

plt.gca().set_box_aspect(1)
plt.xlabel("DeepSet output")
plt.ylabel("density")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(output_dir/"02_hist_output_normal.pdf")
plt.close()

# %%
# Plot the ratio n_Bs/n_Bd like in testing
print("Plot n_Bs/n_Bd...")

Bd_idxs = np.argwhere(y_test==0).flatten()
Bs_idxs = np.argwhere(y_test==1).flatten()
N_Bd = Bd_idxs.shape[0]
N_Bs = int(0.011 * N_Bd)
print(f"N_Bs_before = {Bs_idxs.shape[0]}")
print(f"{N_Bd = }")
print(f"{N_Bs = }")
Bs_idxs_reduced = np.random.choice(Bs_idxs, N_Bs)

idxs_ = np.concatenate([Bd_idxs,Bs_idxs_reduced])
np.random.shuffle(idxs_)

y_test_ratio = y_test[idxs_]
y_pred_proba_test_ratio = y_pred_proba_test[idxs_]

cuts = np.linspace(0,0.85,200)

ratio_lower = []
ratio_greater = []

for cut in cuts:
    idxs_lower = np.argwhere(y_pred_proba_test_ratio <= cut).flatten()
    idxs_greater = np.argwhere(y_pred_proba_test_ratio >= cut).flatten()
    if len(idxs_lower) == 0:
        ratio_lower.append(-1)
    else:
        n_Bd_lower = np.sum(y_test_ratio[idxs_lower] == 0)
        n_Bs_lower = np.sum(y_test_ratio[idxs_lower] == 1)
        ratio_lower.append(n_Bs_lower/n_Bd_lower)
        
    if len(idxs_greater) == 0:
        ratio_greater.append(-1)
    else:
        n_Bd_greater = np.sum(y_test_ratio[idxs_greater] == 0)
        n_Bs_greater = np.sum(y_test_ratio[idxs_greater] == 1)
        ratio_greater.append(n_Bs_greater/n_Bd_greater)

ratio_lower = np.array(ratio_lower)
ratio_greater = np.array(ratio_greater)



# plotting
fig = plt.figure(figsize=(4,4))

plt.axhline(0.011, color="black", linestyle="--", label=f"expected: 0.011")

plt.plot(cuts[ratio_greater>=0], ratio_greater[ratio_greater>=0], "-" , label=f"n_Bs/n_Bd (ProbBs>=cut)")
plt.plot(cuts[ratio_lower>=0], ratio_lower[ratio_lower>=0], "-" , label=f"n_Bs/n_Bd (ProbBs<=cut)")
        
plt.gca().set_box_aspect(1)
plt.xlabel("ProbBs cut")
plt.ylabel("n_Bs / n_Bd")
#plt.xlim(0,0.7)
#plt.yscale("log")
plt.legend()
plt.tight_layout()
#plt.show()
fig.savefig(output_dir/f"03_B_ratio_by_cut.pdf")
plt.close()

# plot only to 0.75
fig = plt.figure(figsize=(4,4))

plt.axhline(0.011, color="black", linestyle="--", label=f"expected: 0.011")

mask_greater = (cuts <= 0.75) & (ratio_greater>=0)
mask_lower = (cuts <= 0.75) & (ratio_lower>=0)
plt.plot(cuts[mask_greater], ratio_greater[mask_greater], "-" , label=f"n_Bs/n_Bd (ProbBs>=cut)")
plt.plot(cuts[mask_lower], ratio_lower[mask_lower], "-" , label=f"n_Bs/n_Bd (ProbBs<=cut)")
        
plt.gca().set_box_aspect(1)
plt.xlabel("ProbBs cut")
plt.ylabel("n_Bs / n_Bd")
#plt.xlim(0,0.7)
#plt.yscale("log")
plt.legend()
plt.tight_layout()
#plt.show()
fig.savefig(output_dir/f"03_B_ratio_by_cut_for_comparison.pdf")
plt.close()



# %%
# Merge all evaluation plots
merge_pdfs(output_dir, paths.B_classifier_eval_plots_file)

# %%
# Save the eval metrics to a file
eval_results = {
    "roc_auc_test" : float(auc),
    "roc_auc_train" : float(auc_train),
    "accuracy_test" : float(accuracy_test),
    "accuracy_train" : float(accuracy_train),
    "confusion_matrix_test" : skmetrics.confusion_matrix(y_test, y_pred_test, normalize="true").tolist()
}

with open(paths.B_classifier_eval_data_file, "w") as file:
    json.dump(eval_results, file, indent=2)

# %%
