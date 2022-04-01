# %%
# Imports
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics

# Imports from this project
from utils import paths
from utils.input_output import load_feature_keys, load_preprocessed_data
from utils.merge_pdfs import merge_pdfs

# %%
# Constant variables

input_file = paths.preprocessed_data_file

# TODO: Save the model
output_file_model = Path("/ceph/users/nguth/models/BDT_SS/test")

output_dir_plots = paths.plots_dir/"SS_classifier_training"
output_dir_plots.mkdir(parents=True, exist_ok=True)

output_file = paths.plots_dir/"eval_ss_classification.pdf"

N_tracks_max = 1000000

load_batch_size = 100000

params = {
    "test_size" : 0.4,
    "n_estimators" : 500,
    "max_depth" : 5,
    "n_threads" : 10,
    "early_stopping_rounds" : 20,
    "scale_pos_weight" : 10,
    "objective" : "binary:logistic",
    "eval_metric" : ["logloss", "error", "auc"]
}

random_seed = 42
rng = np.random.default_rng(random_seed)

# %%
# Read the input data
print("Read in the data...")
df = load_preprocessed_data(N_entries_max=1000000, batch_size=100000)
print("Done reading data")

# %%
# Prepare the data
feature_keys = load_feature_keys(include_keys=["extracted", "direct"], 
                                 exclude_keys=["not_for_training"])

label_key = "Tr_is_SS"

X = df[feature_keys]
y = df[label_key].to_numpy()

# Split the data into train and test (with shuffling)
temp = train_test_split(X, y, 
                          test_size=params["test_size"], 
                          shuffle=True, 
                          stratify=y)
X_train, X_test, y_train, y_test = temp

print(f"Training Tracks: {len(y_train)}")
print(f"Test Tracks: {len(y_test)}")

# %%
# Train a BDT

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

# define the BDT
bdt = xgb.XGBClassifier(n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"], 
                        nthread=params["n_threads"],
                        objective=params["objective"],
                        scale_pos_weight=params["scale_pos_weight"],
                        use_label_encoder=False)

# Train the BDT
bdt.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        eval_metric=params["eval_metric"],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose=0,
        callbacks=[XGBProgressCallback(params["n_estimators"], "BDT Training")])




# %%
###############
# Evaluation
###############
# %%
# Evaluate the training

# Error rate during training
validation_score = bdt.evals_result()
for i, metric in enumerate(params["eval_metric"]):
    iteration = range(len(validation_score["validation_0"][metric]))
    plt.figure(figsize=(8, 6))
    plt.title(f"training performance ({metric})")
    plt.plot(iteration, validation_score["validation_0"][metric], label="training data")
    plt.plot(iteration, validation_score["validation_1"][metric], label="test data")
    plt.xlabel("iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(output_dir_plots/f"00_train_performance_{i}_{metric}.pdf")
    plt.show()

# %%
# Feature Importance
# https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
# Build the dataframe
df_fi = pd.DataFrame({"feature":feature_keys})
importance_types = ["weight", "gain", "total_gain", "cover", "total_cover"]
for imp_type in importance_types:
    scores = bdt.get_booster().get_score(importance_type=imp_type)
    df_fi[imp_type] = df_fi["feature"].map(scores)
    # norm on 1
    df_fi[imp_type] /= df_fi[imp_type].sum()

# Sort the features
df_fi.sort_values(by="gain", ascending=False, inplace=True)

# Show in Text version
#with pd.option_context('display.min_rows',14):
#    print(df_fi)
df_fi.to_csv(output_dir_plots/"feature_importance.csv")

# Plot the Feature importance
fig, axs = plt.subplots(len(importance_types), 1,figsize=(len(feature_keys), len(importance_types)*5), sharex=True)
fig.suptitle("Feature Importance", fontsize=40, y=0.995)
for i, (imp_type, ax) in enumerate(zip(importance_types, axs)):
    ax.bar(df_fi["feature"], df_fi[imp_type], color=f"C{i}",zorder=3)
    ax.set_ylabel(imp_type, fontsize=30)
    ax.grid(zorder=0)
    ax.tick_params(axis="x", labelbottom=True, labelrotation=45)
plt.tight_layout()
plt.savefig(output_dir_plots/"01_feature_importance.pdf")
plt.show()

# %%
# Evaluate the model on test data

# get predictions
y_pred_proba = bdt.predict_proba(X_test)

# %%
# ROC curve
fpr, tpr, _ = skmetrics.roc_curve(y_test, y_pred_proba[:, 1])
auc = skmetrics.auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title(f"ROC curve, AUC={auc:.4f}")
plt.plot(fpr, tpr)
plt.xlabel("False positive rate (background)")
plt.ylabel("True positive rate (sameside)")
plt.savefig(output_dir_plots/"02_roc.pdf")
plt.show()

# %%
# Probability Distribution
plt.figure(figsize=(8,6))
plt.title("Distribution of the probabilities")
plt.hist(y_pred_proba[y_test==0][:,1], histtype="step", bins=200, density=True, label="other (ground truth)")
plt.hist(y_pred_proba[y_test==1][:,1], histtype="step", bins=200, density=True, label="SS (ground truth)")
plt.yscale("log")
plt.xlabel("Prediction Probability of SS")
plt.ylabel("density (logarithmic)")
plt.legend()
plt.savefig(output_dir_plots/"03_hist_proba.pdf")
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

tp,fp,fn,tn = np.apply_along_axis(rates_for_cut, 1,cut_linspace[:,np.newaxis], y_test, y_pred_proba, tqdm(total=len(cut_linspace), desc="Calc tp,fp,fn,tn")).T

# %%
# Plot the rates for every cut
plt.figure(figsize=(8,6))
plt.title("Prediction rates for every cut")
plt.plot(cut_linspace, tp, label="tp (True Positive)", linestyle="dashed")
plt.plot(cut_linspace, fn, label="fn (False Negative)", linestyle="dashed")
plt.plot(cut_linspace, tn, label="tn (True Negative)")
plt.plot(cut_linspace, fp, label="fp (False Positive)")
plt.xlabel("Cut")
#plt.yscale("log")
plt.legend()
plt.savefig(output_dir_plots/"04_pred_rates.pdf")
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
plt.savefig(output_dir_plots/"05_metrics.pdf")
plt.show()

# %%
# Merge all evaluation plots
merge_pdfs(output_dir_plots, output_file)

# %%
