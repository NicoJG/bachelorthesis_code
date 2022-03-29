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
from utils.merge_pdfs import merge_pdfs

# %%
# Constant variables

input_file = Path("/ceph/users/nguth/data/preprocessed_mc_Sim9b.root")

output_file_model = Path("/ceph/users/nguth/models/BDT_SS/test")

output_dir_plots = Path("plots/SS_classifier_training")
output_dir_plots.mkdir(parents=True, exist_ok=True)

output_file = Path("plots/eval_ss_classification_random_features.pdf")

N_tracks_max = 1000000

load_batch_size = 100000

params = {
    "test_size" : 0.4,
    "n_estimators" : 300,
    "max_depth" : 5,
    "n_threads" : 10,
    "early_stopping_rounds" : 20
}

random_seed = 42
rng = np.random.default_rng(random_seed)

# %%
# Read Num of Entries
with uproot.open(input_file)["DecayTree"] as tree:
    N_tracks_in_tree = tree.num_entries

N_tracks = np.min([N_tracks_in_tree, N_tracks_max])

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)

print(f"Tracks in the preprocessed data: {N_tracks_in_tree}")
print(f"Tracks used for training and testing: {N_tracks}")

# %%
# Read the input data
print("Read in the data...")

df = pd.DataFrame()
with uproot.open(input_file)["DecayTree"] as tree:
    tree_iter = tree.iterate(entry_stop=N_tracks, step_size=load_batch_size, library="pd")
    for temp_df in tqdm(tree_iter, "Tracks", total=N_batches_estimate):
        temp_df.set_index("index", inplace=True)
        df = pd.concat([df, temp_df])

print("Done reading input")

# %%
# Prepare the data
with open("features.json") as features_file:
    features_dict = json.load(features_file)
    
feature_keys = []
for k in ["extracted", "direct"]:
    feature_keys.extend(features_dict[k])

for k in features_dict["not_for_training"]:
    feature_keys.remove(k)

label_key = "Tr_is_SS"

X = df[feature_keys]
y = df[label_key].to_numpy()

# Split the data into train and test (with shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=params["test_size"], 
                                                    shuffle=True, 
                                                    stratify=y)

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
                        use_label_encoder=False)

# Train the BDT
bdt.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose=0,
        callbacks=[XGBProgressCallback(params["n_estimators"], "BDT Training")])





###############
# Evaluation
###############
# %%
# Evaluate the training

# Error rate during training
validation_score = bdt.evals_result()
iteration = range(len(validation_score["validation_0"]["logloss"]))
plt.figure(figsize=(8, 6))
plt.title("training performance")
plt.plot(iteration, validation_score["validation_0"]["logloss"], label="Training performance")
plt.plot(iteration, validation_score["validation_1"]["logloss"], label="Test performance")
plt.xlabel("iteration")
plt.ylabel("error rate")
plt.legend()
plt.savefig(output_dir_plots/"train_performance.pdf")
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
plt.savefig(output_dir_plots/"00_roc.pdf")
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
plt.savefig(output_dir_plots/"01_hist_proba.pdf")
plt.show()

# %%
# Analysis of different cuts
cut_linspace = np.linspace(0,1,1000)

tp, tn, fp, fn = [-1*np.ones_like(cut_linspace) for i in range(4)]

# make predictions for every cut and calc the confusion matrix
# TODO: faster
for i, cut in enumerate(tqdm(cut_linspace)):
    conf_mat = skmetrics.confusion_matrix(y_test, y_pred_proba[:,1] >= cut, normalize="all")
    tn[i], fp[i], fn[i], tp[i] = conf_mat.flatten()

# %%
# Plot the rates for every cut
plt.plot(cut_linspace, tp, label="tp (True Positive)", linestyle="dashed")
plt.plot(cut_linspace, fn, label="fn (False Negative)", linestyle="dashed")
plt.plot(cut_linspace, tn, label="tn (True Negative)")
plt.plot(cut_linspace, fp, label="fp (False Positive)")
#plt.yscale("log")
plt.legend()
plt.show()


# %%
# Confusion matrix
conf_mat = skmetrics.confusion_matrix(y_test, y_pred)/len(y_pred)
fig, ax = plt.subplots(figsize=(8,6))
skmetrics.ConfusionMatrixDisplay(conf_mat, display_labels=["other","SS"]).plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig(output_dir_plots/"confusion_matrix.pdf")
plt.show()

tn = conf_mat[0,0]
fp = conf_mat[0,1]
fn = conf_mat[1,0]
tp = conf_mat[1,1]
# %%
# Plot the probability histogram

# %%
# Calculate different metrics (and plot them as text so they are in a pdf)

fig = plt.figure(figsize=(10,6))
plt.axis("off")
plt.text(0.1, 0.1, 
f"""
Various Metrics

Accuracy: {skmetrics.accuracy_score(y_test, y_pred):.4f}
Balanced Accuracy: {skmetrics.balanced_accuracy_score(y_test, y_pred):.4f}
Precision: {skmetrics.precision_score(y_test, y_pred):.4f}
Recall: {skmetrics.recall_score(y_test, y_pred):.4f}

Classification Report:
{skmetrics.classification_report(y_test, y_pred)}
""",
         fontsize=16,
         fontfamily="monospace",
         horizontalalignment="left",
         transform=fig.transFigure)
plt.tight_layout()
plt.savefig(output_dir_plots/"various_metrics.pdf")
plt.show()

# %%
# Feature Importance
df_feature_importance = pd.DataFrame({"feature":feature_keys, "feature_importance":bdt.feature_importances_})
df_feature_importance.sort_values(by="feature_importance", ascending=False, inplace=True)
with pd.option_context('display.min_rows',14):
    print(df_feature_importance)

df_feature_importance.to_csv(output_dir_plots/"feature_importance.csv")

# Plot the Feature importance
plt.figure(figsize=(len(feature_keys), 5))
plt.title("Feature Importance")
plt.bar(df_feature_importance["feature"], df_feature_importance["feature_importance"])
# plt.yscale("log")
plt.ylabel("Feature Importance")
plt.xticks(rotation=45)
plt.grid(which="both")
plt.tight_layout()
plt.savefig(output_dir_plots/"feature_importance.pdf")
plt.show()

# %%
merge_pdfs(output_dir_plots, output_file)

# %%
