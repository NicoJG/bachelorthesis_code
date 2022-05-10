# %%
# Imports
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
import sklearn.metrics as skmetrics
import torch
import shutil

# Local Imports
from utils.input_output import load_data_from_root, load_feature_keys
from utils import paths
from utils.histograms import get_hist
from utils.merge_pdfs import merge_pdfs

# %%
# Constants
parser = ArgumentParser()
parser.add_argument("-t", "--threads", dest="n_threads", default=50, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

N_events = 100000000

batch_size_tracks = 10000
batch_size_events = 100000

assert paths.ss_classifier_model_file.is_file(), f"The model '{paths.ss_classifier_model_file}' does not exist yet."
assert paths.B_classifier_model_file.is_file(), f"The model '{paths.B_classifier_model_file}' does not exist yet."
assert paths.bkg_bdt_model_file.is_file(), f"The model '{paths.bkg_bdt_model_file}' does not exist yet."

data_file = paths.B2JpsiKS_Data_file

data_tree_key = "Bd2JpsiKSDetached/DecayTree"

# %%
# Load all relevant feature keys
# Event features:
bdt_features_data = load_feature_keys(["features_BKG_BDT_data"], file_path=paths.features_data_testing_file)

lambda_veto_features = load_feature_keys(["features_Lambda_cut"], file_path=paths.features_data_testing_file)

features_other_cuts = load_feature_keys(["features_other_cuts"], file_path=paths.features_data_testing_file)

# Track features:
for_feature_extraction = load_feature_keys(["for_feature_extraction"], file_path=paths.features_data_testing_file)

extracted_features = load_feature_keys(["extracted_features"], file_path=paths.features_data_testing_file)

main_features = load_feature_keys(["main_features"], file_path=paths.features_data_testing_file)

features_SS_classifier = load_feature_keys(["features_SS_classifier"], file_path=paths.features_SS_classifier_file)

features_B_classifier = load_feature_keys(["features_B_classifier"], file_path=paths.features_B_classifier_file)

# Load all a list of all existing keys in the data
with uproot.open(data_file)[data_tree_key] as tree:
    data_all_feature_keys = tree.keys()

# Check if all features are in the dataset
event_features_to_load = []
event_features_to_load += bdt_features_data 
event_features_to_load += lambda_veto_features 
event_features_to_load += features_other_cuts
event_features_to_load += main_features

track_features_to_load = []
track_features_to_load += for_feature_extraction
track_features_to_load += features_SS_classifier
track_features_to_load += features_B_classifier
track_features_to_load = list(set(track_features_to_load) - set(extracted_features))

assert len(set(event_features_to_load+track_features_to_load) - set(data_all_feature_keys))==0, f"The following features are not found in the data: {set(event_features_to_load+track_features_to_load) - set(data_all_feature_keys)}"

# %%
# Load the tracks data for applying the B classification
print("Load the tracks data for applying the B classification...")
df_tracks = load_data_from_root(data_file, data_tree_key, 
                              features=track_features_to_load,
                              n_threads=n_threads,
                              N_entries_max=N_events,
                              batch_size=batch_size_tracks)


# %%
# Extract all needed features for the Bd/Bs classification
df_tracks.reset_index(drop=False, inplace=True)
df_tracks.rename(columns={"entry":"event_id","subentry":"track_id"}, inplace=True)

df_tracks['Tr_diff_z'] = df_tracks['Tr_T_TrFIRSTHITZ'] - df_tracks['B_OWNPV_Z']

PX_proj = -1 * df_tracks[f"B_PX"] * df_tracks[f"Tr_T_PX"]
PY_proj = -1 * df_tracks[f"B_PY"] * df_tracks[f"Tr_T_PY"]
PZ_proj = -1 * df_tracks[f"B_PZ"] * df_tracks[f"Tr_T_PZ"]
PE_proj = +1 * df_tracks[f"B_PE"] * df_tracks[f"Tr_T_E"]

df_tracks["Tr_p_proj"] = np.sum([PX_proj, PY_proj, PZ_proj, PE_proj], axis=0)
# del PX_proj, PY_proj, PZ_proj, PE_proj

df_tracks['Tr_diff_pt'] = df_tracks["B_PT"] - df_tracks["Tr_T_PT"]

df_tracks['Tr_diff_p'] = df_tracks["B_P"] - df_tracks["Tr_T_P"]

df_tracks['Tr_cos_diff_phi'] = np.array(list(map(lambda x : np.cos(x), df_tracks['B_LOKI_PHI'] - df_tracks['Tr_T_Phi'])))

df_tracks['Tr_diff_eta'] = df_tracks['B_LOKI_ETA'] - df_tracks['Tr_T_Eta']

# %%
# Load and apply both models for the B classification

print("Apply SS classifier...")
# Load the SS classifier
with open(paths.ss_classifier_model_file, "rb") as file:
    model_SS_classifier = pickle.load(file)
    
# Apply the SS classifier
df_tracks["Tr_ProbSS"] = model_SS_classifier.predict_proba(df_tracks[features_SS_classifier])[:,1]

print("Apply B classifier...")
# Load the B classifier
model_B_classifier = torch.load(paths.B_classifier_model_file, map_location="cpu")

# Apply the B classifier
X = df_tracks[["event_id"]+features_B_classifier].to_numpy()
event_ids = df_tracks["event_id"].unique()
B_ProbBs = model_B_classifier.decision_function(X)

print("Successfully calculated 'B_ProbBs'")

# %%
# Clear the RAM, as only the Output of the B classification matters now
# del df_tracks, X, model_SS_classifier, model_B_classifier

# %%
####################################
# Start the the analysis of the data
####################################

output_dir = paths.data_testing_plots_dir

if paths.data_testing_plots_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir()


# %%
# Load the event data for further analysis of the B classification output
print("Load the event data for further analysis of the B classification output...")
df = load_data_from_root(data_file, data_tree_key, 
                              features=event_features_to_load,
                              n_threads=n_threads,
                              N_entries_max=N_events,
                              batch_size=batch_size_events)

# %%
# Make an event_id column
df.reset_index(drop=False, inplace=True)
df.rename(columns={"index":"event_id"}, inplace=True)


# %%
# Check if the event ids match the event ids used for the B classification
assert np.all(df["event_id"]==event_ids), "There is a mismatch in the event ids"

# Add the B classification as a feature
df["B_ProbBs"] = B_ProbBs

# %%
# Load and Apply the background bdt
print("Apply BKG BDT...")
# Load the SS classifier
with open(paths.bkg_bdt_model_file, "rb") as file:
    model_BKG_BDT = pickle.load(file)
    
# Apply the SS classifier
df["B_BKG_BDT"] = model_BKG_BDT.predict_proba(df[bdt_features_data])[:,1]

# %%
# Prepare the veto of the Lambda Background
print("Make the Lambda veto...")

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
n_bins = 100
x_min, x_max = 1060.0 , 1180.0
bins = np.linspace(x_min, x_max, n_bins+1)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle(r"invariant mass used for the $\Lambda$ veto")

ax0.hist(df["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="no veto")
ax0.hist(df.query("lambda_veto==0")["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="with veto")
ax1.hist(df["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="no veto")
ax1.hist(df.query("lambda_veto==0")["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="with veto")

ax0.set_xlabel(r"$m(\pi^- p^+)$")
ax1.set_xlabel(r"$m(\pi^+ p^-)$")

ax0.set_ylabel("counts")
ax1.set_ylabel("counts")

ax0.legend()
ax1.legend()
plt.tight_layout()
plt.savefig(output_dir/"01_lambda_veto.pdf")
plt.close()

# %%
# Make additional cuts


# %%
# Plot the B mass with and without the cuts and the B classification
print("Create all the B mass plots...")

x_min = 5100. # MeV
x_max = 5600. # MeV

n_bins = 200

bin_edges = np.linspace(x_min, x_max, n_bins+1)

# Plot raw B mass distribution
plt.figure(figsize=(8,6))
plt.title("B mass distribution")
plt.hist(df["B_M"], histtype="step", bins=bin_edges)
plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir/"03_B_mass_00_raw.pdf")
plt.close()

# Plot B mass distribution after bdt cut
df_cut_bdt = df.query("(B_BKG_BDT>=0.5)")

plt.figure(figsize=(8,6))
plt.title("B mass distribution with cut on the bkg bdt")
plt.hist(df_cut_bdt["B_M"], histtype="step", bins=bin_edges)
plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir/"03_B_mass_01_with_bdt_cut.pdf")
plt.close()

# Plot B mass distribution after all cuts
df_cut = df.query("(lambda_veto==0)&(B_BKG_BDT>=0.5)")

plt.figure(figsize=(8,6))
plt.title("B mass distribution with all cuts applied (currently only lambda and bdt)")
plt.hist(df_cut["B_M"], histtype="step", bins=bin_edges)
plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir/"03_B_mass_02_with_all_cuts.pdf")
plt.close()

# Plot B mass distribution after B classification
x = [df_cut.query("B_ProbBs<0.5")["B_M"], df_cut.query("B_ProbBs>=0.5")["B_M"]]
labels = ["Bd (predicted)", "Bs (predicted)"]

plt.figure(figsize=(8,6))
plt.title("B mass distribution with cuts and B classification")
plt.hist(x, histtype="step", bins=bin_edges, label=labels)
plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.savefig(output_dir/"03_B_mass_03_B_classification.pdf")
plt.close()

print("Plotting done.")

# %%
merge_pdfs(output_dir, paths.data_testing_plots_file)

# %%
