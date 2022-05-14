# %%
# Imports
import numpy as np
import pandas as pd
import uproot
from argparse import ArgumentParser
import pickle
import torch
from tqdm import tqdm

# Local Imports
from utils.input_output import load_data_from_root, load_feature_keys
from utils import paths
from utils.histograms import get_hist
#from utils.merge_pdfs import merge_pdfs

# %%
# Constants
parser = ArgumentParser()
parser.add_argument("-t", "--threads", dest="n_threads", default=50, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

N_events = 1000000000000

batch_size_tracks = 100000

assert paths.ss_classifier_model_file.is_file(), f"The model '{paths.ss_classifier_model_file}' does not exist yet."
assert paths.B_classifier_model_file.is_file(), f"The model '{paths.B_classifier_model_file}' does not exist yet."

data_files = paths.B2JpsiKS_data_files

data_tree_key = "Bd2JpsiKSDetached/DecayTree"
data_tree_keys = [data_tree_key]*len(data_files)

# %%
# Load all relevant feature keys
# Track features:
for_feature_extraction = load_feature_keys(["for_feature_extraction"], file_path=paths.features_data_testing_file)

extracted_features = load_feature_keys(["extracted_features"], file_path=paths.features_data_testing_file)

features_SS_classifier = load_feature_keys(["features_SS_classifier"], file_path=paths.features_SS_classifier_file)

features_B_classifier = load_feature_keys(["features_B_classifier"], file_path=paths.features_B_classifier_file)

# Load all a list of all existing keys in the data

# Check if all features are in the dataset
track_features_to_load = []
track_features_to_load += for_feature_extraction
track_features_to_load += features_SS_classifier
track_features_to_load += features_B_classifier
track_features_to_load = list(set(track_features_to_load) - set(extracted_features))

for data_file in data_files:
    with uproot.open(data_file)[data_tree_key] as tree:
        data_all_feature_keys = tree.keys()

    assert len(set(track_features_to_load) - set(data_all_feature_keys))==0, f"The following features are not found in the data: {set(track_features_to_load) - set(data_all_feature_keys)}"

# %%
# Load and Apply to all datasets
output_dfs = []

for i, (data_file, data_tree_key) in enumerate(tqdm(zip(data_files, data_tree_keys), desc="Datasets", total=len(data_files))):
    # Load the tracks data for applying the B classification
    df_tracks = load_data_from_root(data_file, data_tree_key, 
                                    features=track_features_to_load,
                                    n_threads=n_threads,
                                    N_entries_max=N_events,
                                    batch_size=batch_size_tracks)
    
    df_tracks.reset_index(drop=False, inplace=True)
    df_tracks.rename(columns={"entry":"event_id", "subentry":"track_id"}, inplace=True)
    
    # Extract all needed features for the Bd/Bs classification
    df_tracks['Tr_diff_z'] = df_tracks['Tr_T_TrFIRSTHITZ'] - df_tracks['B_OWNPV_Z']

    PX_proj = -1 * df_tracks[f"B_PX"] * df_tracks[f"Tr_T_PX"]
    PY_proj = -1 * df_tracks[f"B_PY"] * df_tracks[f"Tr_T_PY"]
    PZ_proj = -1 * df_tracks[f"B_PZ"] * df_tracks[f"Tr_T_PZ"]
    PE_proj = +1 * df_tracks[f"B_PE"] * df_tracks[f"Tr_T_E"]

    df_tracks["Tr_p_proj"] = np.sum([PX_proj, PY_proj, PZ_proj, PE_proj], axis=0)

    df_tracks['Tr_diff_pt'] = df_tracks["B_PT"] - df_tracks["Tr_T_PT"]

    df_tracks['Tr_diff_p'] = df_tracks["B_P"] - df_tracks["Tr_T_P"]

    df_tracks['Tr_cos_diff_phi'] = np.array(list(map(lambda x : np.cos(x), df_tracks['B_LOKI_PHI'] - df_tracks['Tr_T_Phi'])))

    df_tracks['Tr_diff_eta'] = df_tracks['B_LOKI_ETA'] - df_tracks['Tr_T_Eta']
    
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
    
    output_dfs.append(pd.DataFrame({"file_id":i, "event_id":event_ids, "B_ProbBs":B_ProbBs}))
    
    print("Successfully calculated 'B_ProbBs'")

# %%
# Save the output of the model
with uproot.recreate(paths.data_testing_B_ProbBs_file) as file:
    for file_path, output_df in zip(data_files, output_dfs):
        file[file_path.stem] = output_df
print("Successfully applied and saved B_ProbBs")

# %%
