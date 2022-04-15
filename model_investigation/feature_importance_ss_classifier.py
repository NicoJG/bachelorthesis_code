# %%
# Imports
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pytest import param
from tqdm.auto import tqdm
import json
import xgboost as xgb
from sklearn.model_selection import learning_curve, train_test_split
from sklearn import metrics as skmetrics
from sklearn.inspection import permutation_importance
from argparse import ArgumentParser
import pickle

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.merge_pdfs import merge_pdfs
from utils import paths

# %%
# Constants
N_tracks = 1000000 # how many tracks to use for calculating permutation importance and shap values

parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", help="name of the model directory")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

if "model_name" in args and isinstance(args.model_name, str):
    paths.update_ss_classifier_name(args.model_name)
else:
    paths.update_ss_classifier_name("SS_classifier")
    
output_dir = paths.ss_classifier_dir/"feature_importance_plots"
if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

output_file = paths.ss_classifier_dir/"feature_importance_ss_classifier.pdf"

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
                                 N_entries_max=1000000000)
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

# reduce the amount of tracks used
X_test_reduced = X_test.iloc[:N_tracks]
y_test_reduced = y_test[:N_tracks]

# %%
# Read in the trained model
with open(paths.ss_classifier_model_file, "rb") as file:
    model = pickle.load(file)
    
# %%
# Prepare the Feature Importance DataFrame
df_fi = pd.DataFrame({"feature":feature_keys})
df_fi.set_index("feature", drop=True, inplace=True)

# %%
# Get the XGBoost internal Feature Importances
importance_types = ["weight", "gain", "total_gain", "cover", "total_cover"]
for imp_type in importance_types:
    scores = model.get_booster().get_score(importance_type=imp_type)
    df_fi.reset_index(drop=False, inplace=True)
    df_fi[f"xgb_{imp_type}"] = df_fi["feature"].map(scores)
    df_fi.set_index("feature", drop=True, inplace=True)
    # norm on 1
    # df_fi[f"xgb_{imp_type}"] /= df_fi[f"xgb_{imp_type}"].sum()


# %%
# Permutation Feature Importance through scikit-learn
# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
print("Calculate the permutation feature importance...")
perm_imp_metrics = ["balanced_accuracy", "roc_auc", "f1"]

n_repeats = 10
pbar = tqdm(total=(len(feature_keys)*n_repeats)+1, desc="Permutation Importance")

def ProgressCallback(y_true, y_pred):
    pbar.update(1)
    return 0

scoring_dict = {metric:metric for metric in perm_imp_metrics}
scoring_dict["progress_callback"] = skmetrics.make_scorer(ProgressCallback)

perm_fi = permutation_importance(model, 
                                 X_test_reduced[feature_keys], 
                                 y_test_reduced,
                                 scoring=scoring_dict,
                                 n_repeats=n_repeats,
                                 n_jobs=1)

for metric in perm_imp_metrics:
    df_fi.loc[feature_keys,f"perm_{metric}"] = perm_fi[metric]["importances_mean"]
    df_fi.loc[feature_keys,f"perm_{metric}_std"] = perm_fi[metric]["importances_std"]
    
print("Done calculating the permutation feature importance")

# %%
# Sort the features
df_fi.sort_values(by="perm_balanced_accuracy", ascending=False, inplace=True)

# %%
# Plot the feature importances
importance_metrics = ["xgb_gain", "xgb_cover", "perm_balanced_accuracy", "perm_roc_auc", "perm_f1"]

fig, axs = plt.subplots(len(importance_metrics),1, 
                        figsize=(len(feature_keys)/1.5, len(importance_metrics)*5), 
                        sharex=True)

#fig.suptitle(f"Feature Importance")

for i, (ax, metric) in enumerate(zip(axs, importance_metrics)):
    ax.set_title(f"feature importance metric: {metric}")
    if f"{metric}_std" in df_fi.columns:
        yerr = df_fi[f"{metric}_std"]
    else:
        yerr = None
    ax.bar(df_fi.index, df_fi[metric], yerr=yerr, color=f"C{i}", alpha=0.8)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelbottom=True, labelrotation=60)

plt.tight_layout()
plt.savefig(output_dir/"00_selected_importances.pdf")
plt.close()

# %%
# Merge all PDFs
merge_pdfs(output_dir,output_file)

# %%
