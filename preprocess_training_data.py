# %%
# Imports
from pathlib import Path
import json
import uproot
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# %%
# Constant variables
input_files = [
    "/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root",
    "/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root"]

output_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")
if not output_file.parent.is_dir():
    output_file.parent.mkdir(parents=True)
    print(f"Created output directory '{output_file.parent.absolute()}'")

N_events_max_per_dataset = 1000000

load_batch_size = 10000


random_seed = 13579
rng = np.random.default_rng(random_seed)

# %%
# Read in the feature keys from features.json and get a list of features that should be read from the input
with open("features.json") as features_file:
    feature_keys = json.load(features_file)

features_to_load = []
for load_key in ["temporary_mc", "direct_mc", "temporary", "direct"]:
    features_to_load.extend(feature_keys[load_key])

# Check if all keys are present in all datasets
for i, input_file_path in enumerate(input_files):
    with uproot.open(input_file_path)["DecayTree"] as tree:
        keys_not_found = set(features_to_load) - set(tree.keys())
        assert len(keys_not_found) == 0, f"Keys not in tree {i}: {keys_not_found}"

# %%
# Read in the number of entries in all datasets
N_events = []
for i, input_file_path in enumerate(input_files):
    with uproot.open(input_file_path)["DecayTree"] as tree:
        N_events.append(tree.num_entries)

N_events_per_dataset = np.min(N_events + [N_events_max_per_dataset])

N_batches_estimate = np.ceil(N_events_per_dataset / load_batch_size).astype(int)

print(f"Events in the datasets: {N_events}")
print(f"Events to load per dataset: {N_events_per_dataset}")

# %%
# Read the data and merge all datasets
print("Read and merge the data...")

# concatenate all DataFrames into one
df = pd.DataFrame()
# iterate over all input files
for i, input_file_path in enumerate(tqdm(input_files, "Datasets")):
    with uproot.open(input_file_path)["DecayTree"] as tree:   
        # iterate over all batches
        tree_iter = tree.iterate(features_to_load, 
                                 entry_stop=N_events_per_dataset,
                                 step_size=load_batch_size, 
                                 library="pd")
        for temp_df in tqdm(tree_iter, f"Batches in File {i}", total=N_batches_estimate):

            temp_df.rename_axis(index={"entry":"event_id", "subentry": "track_id"},  inplace=True)

            temp_df["input_file_id"] = i

            if "B2JpsiKstar" in input_file_path:
                temp_df["decay_id"] = 0
            elif "Bs2DsPi" in input_file_path:
                temp_df["decay_id"] = 1
            else:
                raise NameError(f"Decay channel not recognized in Dataset {i}")

            # make sure all event ids are unambiguous
            if not df.empty:
                temp_df.reset_index(inplace=True)
                temp_df["event_id"] += df.index.max()[0] + 1
                temp_df.set_index(["event_id", "track_id"], inplace=True)

            # append this batch to the DataFrame
            df = pd.concat([df, temp_df])

print("Done reading input")

# %%
# Selection of MC events
print("Preselection...")

N_events_before_sel = len(df.index.unique("event_id"))

# We only want B_BKGCAT == 0 and B_BKGCAT == 50
mask = (df["B_BKGCAT"] == 0) | (df["B_BKGCAT"] == 50)
# but B_BKGCAT is wrong in Bs2DsPi... workaround:
mask |= (df["decay_id"] == 1) & (df["B_BKGCAT"] == 20)
# adjust for different selection parameters in both datasets
mask &= df["B_DIRA_OWNPV"] > 0.9999
mask &= df["B_IPCHI2_OWNPV"] < 16

df = df[mask]

N_events_after_sel = len(df.index.unique("event_id"))

print(f"Events before selection: {N_events_before_sel}")
print(f"Events after selection: {N_events_after_sel}")

# %%
# Shuffle the events
print("Shuffle all events...")

df_len_before_shuffle = len(df)

old_event_ids = df.index.unique("event_id")
shuffled_event_ids = rng.permutation(old_event_ids)

df = df.loc[shuffled_event_ids]

assert len(df) == df_len_before_shuffle, "Shuffling shrunk the dataframe. Something went wrong."

# %%
# Fix the event ids
print("Reindex all events ascending...")

df_len_before_reindex = len(df)

old_event_ids = df.index.unique("event_id")
reindexed_event_ids = np.arange(len(old_event_ids))
mapping = {old:new for old, new in zip(old_event_ids, reindexed_event_ids)}

# needed for editing the event_id column:
df.reset_index(inplace=True)

df["event_id"] = df["event_id"].map(mapping)

df.set_index(["event_id", "track_id"], inplace=True)

assert len(df) == df_len_before_reindex, "Reindexing shrunk the dataframe. Something went wrong."

# %%
# Generate new features
print("Generate new features...")

def constructVariables(df):

    df['Tr_diff_z'] = df['Tr_T_TrFIRSTHITZ'] - df['B_OWNPV_Z']
    
    PX_proj = -1 * df[f"B_PX"] * df[f"Tr_T_PX"]
    PY_proj = -1 * df[f"B_PY"] * df[f"Tr_T_PY"]
    PZ_proj = -1 * df[f"B_PZ"] * df[f"Tr_T_PZ"]
    PE_proj = +1 * df[f"B_PE"] * df[f"Tr_T_E"]

    df["Tr_p_proj"] = np.sum([PX_proj, PY_proj, PZ_proj, PE_proj], axis=0)

    df['Tr_diff_pt'] = df["B_PT"] - df["Tr_T_PT"]

    df['Tr_diff_p'] = df["B_P"] - df["Tr_T_P"]

    df['Tr_cos_diff_phi'] = np.array(list(map(lambda x : np.cos(x), df['B_LOKI_PHI'] - df['Tr_T_Phi'])))

    df['Tr_diff_eta'] = df['B_LOKI_ETA'] - df['Tr_T_Eta']

    df["Tr_is_SS"] = (df["Tr_ORIG_FLAGS"] == 1).astype(int)

    assert set(df["B_TRUEID"].unique()) == set([511,-511,531,-531]), "There are other signal particles than B0 and Bs"

    df["B_is_strange"] = (np.abs(df["B_TRUEID"]) == 531).astype(int)

    return df

df = constructVariables(df)

# %%
# Remove all temporary features
print("Remove temporary features...")

features_to_remove = []
for remove_key in ["temporary_mc", "temporary"]:
    features_to_remove.extend(feature_keys[remove_key])

df.drop(columns=features_to_remove, inplace=True)

# Check if only the features listed in features.json as direct or extracted are present
features_promised = []
for promised_key in ["extracted_mc", "direct_mc", "extracted", "direct"]:
    features_promised.extend(feature_keys[promised_key])

features_assym = set(df.columns) ^ set(features_promised)

assert len(features_assym) == 0, f"Found an assymmetry in the final features with features.json: {features_assym}"

# %%
# Reorder the features according to the features.json
df = df[features_promised]

# %%
# Save the dataframe to a root file
print("Writing output file...")

with uproot.recreate(output_file) as file:
    file["DecayTree"] = df.reset_index()
    
print("Done writing output")

# %%
