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
from sklearn.metrics import roc_curve

# %%
# Constant variables

input_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")

N_tracks = 100000

load_batch_size = 10000

random_seed = 13579
rng = np.random.default_rng(random_seed)

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)

if not Path("plots").is_dir():
    Path("plots").mkdir()

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

feature_keys.remove("input_file_id")

label_key = "Tr_is_SS"

X = df[feature_keys]
y = df[label_key]

# Split the data into train and test (with shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.4, 
                                                    shuffle=True, 
                                                    stratify=y)

# %%
# Train a BDT
bdt = xgb.XGBClassifier(n_estimators=1000, max_depth=5, n_threads=10)

bdt.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=20)

# %%
# Evaluate the training

# ROC curve
fpr, tpr, _ = roc_curve(y_test, bdt.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.title("ROC curve")
plt.plot(fpr, tpr)
plt.xlabel("False positive rate (background)")
plt.ylabel("True positive rate (sameside)")
plt.savefig("plots/roc.pdf")
plt.show()

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
plt.savefig("plots/train_performance.pdf")
plt.show()

# Feature Importance
df_feature_importance = pd.DataFrame({"feature":feature_keys, "feature_importance":bdt.feature_importances_})
df_feature_importance.sort_values(by="feature_importance", ascending=False, inplace=True)
df_feature_importance

# %%
