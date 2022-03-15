# %%
# Imports
import pathlib
import json
from matplotlib.style import library
import uproot
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# %%
# Constant variables
input_files = [
    "/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root",
    "/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root"]

output_file = "/ceph/users/nguth/data/preprocesses_mc_Sim9b.root"

N_events_per_dataset = 10000

load_batch_size = 1000


# %%
# Read in feature keys
with open("features.json") as features_file:
    feature_keys = json.load(features_file)

print("Features:")
print(feature_keys)

load_keys = ["temporary", "direct", "temporary_mc", "direct_mc"]
features_to_load = []
for load_key in load_keys:
    features_to_load.extend(feature_keys[load_key])

# %%
# Check if all keys are present in all datasets
with (uproot.open(input_files[0])["DecayTree"] as tree0,
     uproot.open(input_files[1])["DecayTree"] as tree1):
    trees = [tree0, tree1]
    for i,tree in enumerate(trees):
        # find specific keys (debugging)
        # print([k for k in tree.keys() if "PV" in k and "IP" in k and "CHI2" in k])
        # check if all keys are found
        keys_not_found = set(features_to_load) - set(tree.keys())
        if len(keys_not_found) > 0:
            raise KeyError(f"Keys not in tree {i}: {keys_not_found}")
    
print("All features found in all trees")
# %%
# Read the data
print("Reading data...")
with (uproot.open(input_files[0])["DecayTree"] as tree0,
     uproot.open(input_files[1])["DecayTree"] as tree1):
    trees = [tree0, tree1]
    # merge datasets
    df = pd.DataFrame()
    for i,tree in enumerate(tqdm(trees, "Datasets")):
        N_batches_estimate = N_events_per_dataset // load_batch_size
        for temp_df in tqdm(tree.iterate(features_to_load, entry_stop=N_events_per_dataset, step_size=load_batch_size, library="pd"), "Batches", total=N_batches_estimate):
            temp_df.reset_index(inplace=True)
            temp_df.rename(columns={"entry":"event_id", "subentry": "track_id"}, errors="raise", inplace=True)
            temp_df["tree_id"] = i
            if not df.empty:
                temp_df["event_id"] = temp_df["event_id"] + df["event_id"].max()

            df = pd.concat([df, temp_df], ignore_index=True)
print("Done.")
# %%
# Selection of MC events
N_events_before_sel = len(df['event_id'].unique())

mask = (df["B_BKGCAT"] == 0) | (df["B_BKGCAT"] == 50)
mask = mask & (df["B_DIRA_OWNPV"] > 0.9999)
mask = mask & (df["B_IPCHI2_OWNPV"] < 16)
df = df[mask]

N_events_after_sel = len(df['event_id'].unique())

print(f"Events before selection: {N_events_before_sel}")
print(f"Events after selection: {N_events_after_sel}")
# %%
