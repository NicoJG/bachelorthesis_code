# %%
# Imports
import sys
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

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.merge_pdfs import merge_pdfs
from utils import paths

# %%
# Constants
output_dir = paths.plots_dir/"feature_importance_ss_classifier"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = paths.plots_dir/"feature_importance_ss_classifier.pdf"

params = {
    "test_size" : 0.4,
    "n_estimators" : 500,
    "max_depth" : 4,
    "learning_rate" : 0.15, # 0.3 is default
    "n_threads" : 50,
    "early_stopping_rounds" : 20,
    #"scale_pos_weight" : 10,
    "objective" : "binary:logistic",
    "eval_metrics" : ["logloss", "map", "error", "auc"]
}

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["label_ss_classifier","features_ss_classifier"])

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(features=feature_keys, 
                                 N_entries_max=1000000)
print("Done reading input")


# %%
# Prepare the data
label_key = "Tr_is_SS"
feature_keys.remove(label_key)

X = df_data[feature_keys]
y = df_data[label_key].to_numpy()

# Split the data into train and test (with shuffling)
temp = train_test_split(X, y, 
                        test_size=params["test_size"], 
                        shuffle=True, 
                        stratify=y)
X_train, X_test, y_train, y_test = temp

print(f"Training Tracks: {len(y_train)}")
print(f"Test Tracks: {len(y_test)}")

# %%
# Build the BDT
params["scale_pos_weight"] = np.sum(y == 0)/np.sum(y == 1)

print(f"scale_pos_weight = {params['scale_pos_weight']}")

model = xgb.XGBClassifier(n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"], 
                    learning_rate=params["learning_rate"],
                    nthread=params["n_threads"],
                    objective=params["objective"],
                    scale_pos_weight=params["scale_pos_weight"],
                    use_label_encoder=False)

# Callback for a progress bar of the training
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

# %%
# Train with all features as a baseline
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          eval_metric=params["eval_metrics"],
          early_stopping_rounds=params["early_stopping_rounds"],
          verbose=0,
          callbacks=[XGBProgressCallback(rounds=params["n_estimators"], 
                                         desc="BDT Train Baseline")]
          )

df_fi = pd.DataFrame({"feature":["_baseline_"]+feature_keys})
df_fi.set_index("feature",drop=True, inplace=True)

temp_results = model.evals_result()["validation_1"]
for metric in params["eval_metrics"]:
    df_fi.loc["_baseline_",metric] = temp_results[metric][model.best_iteration]
df_fi.loc["_baseline_","best_iter"] = model.best_iteration

# %%
# Get the XGBoost Feature Importances
importance_types = ["weight", "gain", "total_gain", "cover", "total_cover"]
for imp_type in importance_types:
    scores = model.get_booster().get_score(importance_type=imp_type)
    df_fi.reset_index(drop=False, inplace=True)
    df_fi[f"xgb_{imp_type}"] = df_fi["feature"].map(scores)
    df_fi.set_index("feature", drop=True, inplace=True)
    # norm on 1
    # df_fi[f"xgb_{imp_type}"] /= df_fi[f"xgb_{imp_type}"].sum()


# %%
# Permutation Feature Importance through scikit-learn on the baseline model
# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
print("Calculate the permutation feature importance...")
perm_imp_metrics = ["balanced_accuracy", "roc_auc"]
perm_fi = permutation_importance(model, X_test[feature_keys], y_test,
                                 scoring=perm_imp_metrics,
                                 n_repeats=50,
                                 n_jobs=params["n_threads"])

for metric in perm_imp_metrics:
    df_fi.loc[feature_keys,f"perm_{metric}"] = perm_fi[metric]["importances_mean"]
    
print("Done calculating the permutation feature importance")

# %%
# Train N times with one feature removed per iteration
for fkey in tqdm(feature_keys, desc="Train without one feature"):
    mask = feature_keys.copy()
    mask.remove(fkey)
    
    model.fit(X_train[mask], y_train, 
              eval_set=[(X_train[mask], y_train), (X_test[mask], y_test)], 
              eval_metric=params["eval_metrics"],
              early_stopping_rounds=params["early_stopping_rounds"],
              verbose=0
              )
    
    temp_results = model.evals_result()["validation_1"]
    for metric in params["eval_metrics"]:
        df_fi.loc[fkey,metric] = temp_results[metric][model.best_iteration]
    df_fi.loc[fkey,"best_iter"] = model.best_iteration

# Calculate the differences to the baseline
for metric in params["eval_metrics"]:
    df_fi[f"diff_{metric}"] = df_fi[metric] - df_fi.loc["_baseline_",metric]

# %%
# Sort the features
df_fi.sort_values(by="diff_logloss", ascending=False, inplace=True)

# %%
# Plot the differences to the baseline sorted
fig, axs = plt.subplots(len(params["eval_metrics"])+1,1, 
                        figsize=(len(feature_keys)/3, (len(params["eval_metrics"])+1)*5), 
                        sharex=True)

fig.suptitle("Feature Importance through temporary feature removal\ndifference in the metrics (removed feature - baseline)")

for i, (ax, metric) in enumerate(zip(axs, params["eval_metrics"])):
    ax.set_title(f"metric: {metric}\nbaseline value: {df_fi.loc['_baseline_',metric]:.6f}")
    ax.bar(df_fi.index, df_fi[f"diff_{metric}"], color=f"C{i}")
    ax.set_ylabel(f"diff_{metric}")
    ax.tick_params(axis="x", labelbottom=True, labelrotation=90)

axs[-1].set_title("Index of the best iteration")
axs[-1].bar(df_fi.index, df_fi["best_iter"], color=f"C{i+1}")
axs[-1].set_ylabel("best_iter")
axs[-1].tick_params(axis="x", labelbottom=True, labelrotation=90)

plt.tight_layout()
plt.savefig(output_dir/"01_removal_importance.pdf")
plt.show()

# %%
# Plot the permutation importance
fig, axs = plt.subplots(len(perm_imp_metrics),1, 
                        figsize=(len(feature_keys)/3, len(perm_imp_metrics)*5), 
                        sharex=True)

fig.suptitle("Permutation Importance from scikit-learn")

for i, (ax, metric) in enumerate(zip(axs, perm_imp_metrics)):
    ax.set_title(f"metric: {metric}")
    ax.bar(df_fi.index, df_fi[f"perm_{metric}"], color=f"C{i}")
    ax.set_ylabel(f"perm_{metric}")
    ax.tick_params(axis="x", labelbottom=True, labelrotation=90)

plt.tight_layout()
plt.savefig(output_dir/"02_permutation_importance.pdf")
plt.show()

# %%
# Plot the XGBoost feature importance
fig, axs = plt.subplots(len(importance_types),1, 
                        figsize=(len(feature_keys)/3, len(importance_types)*6), 
                        sharex=True)

fig.suptitle("XGBoost Internal Feature Importance")

for i, (ax, metric) in enumerate(zip(axs, importance_types)):
    ax.set_title(f"metric: {metric}")
    ax.bar(df_fi.index, df_fi[f"xgb_{metric}"], color=f"C{i}")
    ax.set_ylabel(f"xgb_{metric}")
    ax.tick_params(axis="x", labelbottom=True, labelrotation=90)

plt.tight_layout()
plt.savefig(output_dir/"03_xgb_importance.pdf")
plt.show()

# %%
# Merge all PDFs
merge_pdfs(output_dir,output_file)

# %%
