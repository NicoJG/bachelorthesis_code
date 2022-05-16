# %%
# Imports
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import xgboost as xgb
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
import sklearn.metrics as skmetrics

# Local Imports
from utils.input_output import load_and_merge_from_root, load_feature_keys
from utils import paths

# %%
# Constants
parser = ArgumentParser()
parser.add_argument("-n", "--model_name", dest="model_name", default="BKG_BDT", help="name of the model directory")
parser.add_argument("-t", "--threads", dest="n_threads", default=50, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

model_name = args.model_name
paths.update_bkg_bdt_name(model_name)

assert not paths.bkg_bdt_model_file.is_file(), f"The model '{paths.bkg_bdt_model_file}' already exists! To overwrite it please (re-)move this directory or choose another model name with the flag '--model_name'."

mc_files = paths.B2JpsiKS_mc_files
data_files = paths.B2JpsiKS_data_files

mc_tree_key = "Bd2JpsiKSDetached/DecayTree"
data_tree_key = "Bd2JpsiKSDetached/DecayTree"

mc_tree_keys = [mc_tree_key]*len(mc_files)
data_tree_keys = [data_tree_key]*len(data_files)

N_events_per_dataset = 1000000000000

params = {
    "test_size" : 0.4,
    "model_params" : {
        "n_estimators" : 2000,
        "max_depth" : 4,
        "learning_rate" : 0.3, # 0.3 is default
        "scale_pos_weight" : "TO BE SET", # sum(negative instances) / sum(positive instances)
        "objective" : "binary:logistic",
        "nthread" : n_threads,
        "tree_method" : "hist",
        #"num_parallel_tree" : 1
    },
    "train_params" : {
        "early_stopping_rounds" : 50,
        "eval_metric" : ["logloss", "error", "auc"],
        "verbose" : 0,
    }
    }


# %%
# Load all relevant feature keys
bdt_features = load_feature_keys(["features_BKG_BDT"], file_path=paths.features_data_testing_file)

lambda_veto_features = load_feature_keys(["features_Lambda_cut"], file_path=paths.features_data_testing_file)


# %%
# Find keys in the Trees
print("Save all feature keys present in each file...")
mc_keys_dict = {}
for i,mc_file in enumerate(mc_files):
    with uproot.open(mc_file)[mc_tree_key] as tree:
        mc_keys_dict[f"file{i}"] = tree.keys()
        
with open(paths.internal_base_dir/"temp"/"mc_keys.json", "w") as file:
    json.dump(mc_keys_dict, file, indent=2)

data_keys_dict = {}
for i,data_file in enumerate(data_files):
    with uproot.open(data_file)[data_tree_key] as tree:
        data_keys_dict[f"file{i}"] = tree.keys()
        
with open(paths.internal_base_dir/"temp"/"data_keys.json", "w") as file:
    json.dump(data_keys_dict, file, indent=2)

# %%
# Load the data for the BDT
print("Load the MC data...")
df_mc = load_and_merge_from_root(mc_files, mc_tree_keys, 
                                features=bdt_features+lambda_veto_features, 
                                cut="B_BKGCAT==0",
                                n_threads=n_threads,
                                N_entries_max_per_dataset=N_events_per_dataset)

print("Load the Data...")
df_data = load_and_merge_from_root(data_files, data_tree_keys, 
                                features=bdt_features+lambda_veto_features, 
                                cut="B_M>5450", 
                                n_threads=n_threads,
                                N_entries_max_per_dataset=N_events_per_dataset)

print(f"Events in MC: {len(df_mc)}")
print(f"Events in data: {len(df_data)}")

# %%
assert set(df_data.columns) == set(df_mc.columns)

# %%
# Create the label key
label_key = "B_is_signal"
df_data[label_key] = 0
df_mc[label_key] = 1

# %%
# Save the feature keys
params["label_key"] = label_key
params["feature_keys"] = bdt_features

# %%
# Merge mc and data
df = pd.concat([df_data, df_mc], ignore_index=True)

# %%
# Prepare the cut of the Lambda Background
m_pi = 139.57039 # MeV
m_p = 938.272081 # MeV

def weird_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    return ( (E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2 )**0.5

E_piplus = (df["piplus_P"]**2 + m_pi**2)**0.5
E_pplus = (df["piplus_P"]**2 + m_p**2)**0.5
E_piminus = (df["piminus_P"]**2 + m_pi**2)**0.5
E_pminus = (df["piminus_P"]**2 + m_p**2)**0.5

df["m_pi-p+"] = weird_mass(E_piminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_pplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])
df["m_p-pi+"] = weird_mass(E_pminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_piplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])

# %%
# Make a veto feature
veto_m_min, veto_m_max = 1100., 1138. 
veto_probnn = 0.05

df["lambda_veto"] = (veto_m_min < df["m_pi-p+"]) & (df["m_pi-p+"] < veto_m_max) & (df["piplus_ProbNNp"] > veto_probnn)
df["lambda_veto"] |= (veto_m_min < df["m_p-pi+"]) & (df["m_p-pi+"] < veto_m_max) & (df["piminus_ProbNNp"] > veto_probnn)

df["lambda_veto"] = df["lambda_veto"].astype(int)

# %%
# Plot the invariant masses for the cut of the Lambda Background
# n_bins = 100
# x_min, x_max = 1060.0 , 1180.0
# bins = np.linspace(x_min, x_max, n_bins+1)
# 
# fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
# ax0.hist(df["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="m_pi-p+")
# ax0.hist(df.query("lambda_veto==0")["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="m_pi-p+ veto")
# ax1.hist(df["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="m_pi+p-")
# ax1.hist(df.query("lambda_veto==0")["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="m_pi+p- veto")
# 
# ax0.legend()
# ax1.legend()
# plt.show()

# %%
# Prepare the data for the BDT training
idxs = df.query("lambda_veto==0").index
idxs_train, idxs_test = train_test_split(idxs, test_size=0.4, shuffle=True, stratify=df.loc[idxs,label_key])

X_train = df.loc[idxs_train,bdt_features]
y_train = df.loc[idxs_train,label_key]

X_test = df.loc[idxs_test,bdt_features]
y_test = df.loc[idxs_test,label_key]

# %%
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
# Train a BDT# Parameters of the model
scale_pos_weight = np.sum(y_train == 0)/np.sum(y_test == 1)
params["model_params"]["scale_pos_weight"] = scale_pos_weight
print(f"scale_pos_weight = {scale_pos_weight}")

model = xgb.XGBClassifier(**params["model_params"], use_label_encoder=False)

model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          **params["train_params"],
          callbacks=[XGBProgressCallback(rounds=params["model_params"]["n_estimators"], desc="BDT Train")]
          )
    
# %%
# Save everything
# Save the indices of the train test split
train_idxs = X_train.index.to_list()
test_idxs = X_test.index.to_list()

paths.bkg_bdt_dir.mkdir(parents=True, exist_ok=True)

with open(paths.bkg_bdt_train_test_split_file, "w") as file:
    json.dump({"train_idxs":train_idxs,"test_idxs":test_idxs}, 
              fp=file, 
              indent=2)
    
# Save the parameters
with open(paths.bkg_bdt_parameters_file, "w") as file:
    json.dump(params, file, indent=2)
    
# Save the model
with open(paths.bkg_bdt_model_file, "wb") as file:
    pickle.dump(model, file)
    
print("Successfully trained and saved the BKG BDT.")
# %%
