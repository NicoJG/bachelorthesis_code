# %%
# Imports
import sys
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
from sklearn.inspection import permutation_importance

# Imports from this project
sys.path.insert(0, Path(__file__).parent.parent.parent)
from utils import paths
from utils.input_output import load_feature_keys, load_preprocessed_data

# %%
# Constant variables

input_file = paths.preprocessed_data_file

output_dir = paths.plots_dir/"SS_classifier_importance"
output_dir.mkdir(parents=True, exist_ok=True)

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
df = load_preprocessed_data(N_entries_max=1000000000,
                            batch_size=100000)
print("Done reading input")

# %%
# Prepare the data
feature_keys = load_feature_keys(["extracted", "direct"], exclude_keys=["not_for_training"])

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
# Check for feature correlation
df_corr = X.corr()
N_features = len(X.columns)
plt.figure(figsize=(N_features/4,N_features/4))
plt.title("Feature Correlation")
ax_img = plt.matshow(df_corr, vmin=-1, vmax=+1, fignum=0, cmap="seismic")
plt.xticks(ticks=range(N_features), labels=X.columns, rotation=90)
plt.yticks(ticks=range(N_features), labels=X.columns)

# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ax_img, cax=cax)

plt.tight_layout()
plt.savefig(output_dir/"feature_correlation.pdf")
plt.show()

# %%
# List the highest correlations

temp_df = df_corr.copy()
# set lower triangular matrix (with diagonal) to nan
triu_mask = np.triu(np.ones(temp_df.shape), k=1).astype(bool)
temp_df.where(triu_mask, np.nan, inplace=True)
# set all abs values lower than ... to nan
temp_df.where(np.abs(temp_df)>0.9, np.nan, inplace=True)
# list of all pairs that have a high correlation
temp_df.dropna(how="all", inplace=True)
temp_df = temp_df.stack().reset_index()
# print and save the list
temp_df.rename(columns={"level_0":"f0", "level_1":"f1", 0:"correlation"}, inplace=True)
temp_df.sort_values(by="correlation", ascending=False, inplace=True)
temp_df = temp_df[["correlation", "f0", "f1"]]
print(temp_df)
temp_df.to_csv(output_dir/"feature_correlation.csv", float_format="%.4f", index=False)

# TODO: check for outliers which could distort the correlation coefficient

# %%
# Create the feature importance Dataframe
# to which the scores are inserted
df_fi = pd.DataFrame({"feature":feature_keys})
# Feature Importance from XGBoost
# https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
importance_types = ["weight", "gain", "total_gain", "cover", "total_cover"]
for imp_type in importance_types:
    scores = bdt.get_booster().get_score(importance_type=imp_type)
    df_fi[imp_type] = df_fi["feature"].map(scores)
    # norm on 1
    df_fi[imp_type] /= df_fi[imp_type].sum()

# %%
# Permutation Feature Importance through scikit-learn
# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
perm_imp_metrics = ["balanced_accuracy", "roc_auc", "accuracy", "precision", "recall"]
perm_fi = permutation_importance(bdt, X_test, y_test,
                                 scoring=perm_imp_metrics,
                                 n_repeats=10,
                                 n_jobs=10)

for metric in perm_imp_metrics:
    df_fi[f"perm_{metric}"] = perm_fi[metric]["importances_mean"]

# %%
# Show the results

# Sort the features
df_fi.set_index("feature", drop=True, inplace=True)
df_fi.sort_values(by="perm_roc_auc", ascending=False, inplace=True)

# Plot the Feature importance
fig, axs = plt.subplots(len(df_fi.columns), 1, figsize=(len(df_fi.index), len(df_fi.columns)*4), sharex=True)
fig.suptitle("Feature Importance", fontsize=40, y=0.995)
for i, (col, ax) in enumerate(zip(df_fi.columns, axs)):
    ax.bar(df_fi.index, df_fi[col], color=f"C{i}",zorder=3)
    ax.set_ylabel(col)
    ax.grid(zorder=0)
    ax.tick_params(axis="x", labelbottom=True, labelrotation=45)
plt.tight_layout()
plt.savefig(output_dir/"feature_importance.pdf")
plt.show()
# %%
