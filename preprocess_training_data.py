# %%
# Imports
from pathlib import Path
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

output_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")
if not output_file.parent.is_dir():
    output_file.parent.mkdir(parents=True)
    print(f"Created output directory '{output_file.parent.absolute()}'")

N_events_per_dataset = 1000

load_batch_size = 10000

N_batches_estimate = N_events_per_dataset // load_batch_size

random_seed = 13579
rng = np.random.default_rng(random_seed)

# %%
# Read in the feature keys
with open("features.json") as features_file:
    feature_keys = json.load(features_file)

# print("Features:")
# print(feature_keys)

load_keys = ["temporary", "direct", "temporary_mc", "direct_mc"]
features_to_load = []
for load_key in load_keys:
    features_to_load.extend(feature_keys[load_key])

# %%
# Check if all keys are present in all datasets
for i, input_file_path in enumerate(input_files):
    with uproot.open(input_file_path)["DecayTree"] as tree:
        # find specific keys (debugging)
        # print([k for k in tree.keys() if "PV" in k and "IP" in k and "CHI2" in k])
        # check if all keys are found
        keys_not_found = set(features_to_load) - set(tree.keys())
        if len(keys_not_found) > 0:
            raise KeyError(f"Keys not in tree {i}: {keys_not_found}")
    
print("All features found in all trees")

# %%
# Read the data and merge all datasets
print("Reading data...")
df = pd.DataFrame()
for i, input_file_path in enumerate(tqdm(input_files, "Datasets")):
    with uproot.open(input_file_path)["DecayTree"] as tree:   
        tree_iter = tree.iterate(features_to_load, 
                                 entry_stop=N_events_per_dataset,
                                 step_size=load_batch_size, 
                                 library="pd")
        for temp_df in tqdm(tree_iter, f"Batches in File {i}", total=N_batches_estimate):

            temp_df.rename_axis(index={"entry":"event_id", "subentry": "track_id"},  inplace=True)

            temp_df["tree_id"] = i

            if "B2JpsiKstar" in input_file_path:
                temp_df["decay"] = "B2JpsiKstar"
            elif "Bs2DsPi" in input_file_path:
                temp_df["decay"] = "Bs2DsPi"
            else:
                raise NameError(f"Decay channel not recognized in Dataset {i}")

            if not df.empty:
                temp_df.reset_index(inplace=True)
                temp_df["event_id"] += df.index.max()[0] + 1
                temp_df.set_index(["event_id", "track_id"], inplace=True)

            df = pd.concat([df, temp_df])

print("Done reading in.")

# %%
# Histogram some features before selection (debug)
# df.hist("B_BKGCAT", by="decay")
# print(df[["B_BKGCAT","decay"]].value_counts())
# df.hist("B_BKGCAT")
# df.hist("B_DIRA_OWNPV")
# df.hist("B_IPCHI2_OWNPV")

# %%
# Selection of MC events
N_events_before_sel = len(df.index.unique("event_id"))

# We only want B_BKGCAT == 0 and B_BKGCAT == 50
mask = (df["B_BKGCAT"] == 0) | (df["B_BKGCAT"] == 50)
# but B_BKGCAT is wrong in Bs2DsPi... workaround:
mask |= (df["decay"] == "Bs2DsPi") & (df["B_BKGCAT"] == 20)
mask &= df["B_DIRA_OWNPV"] > 0.9999
mask &= df["B_IPCHI2_OWNPV"] < 16

df = df[mask]

N_events_after_sel = len(df.index.unique("event_id"))

print(f"Events before selection: {N_events_before_sel}")
print(f"Events after selection: {N_events_after_sel}")

# %%
# Shuffle the events
event_ids = df.index.unique("event_id")

shuffled_event_ids = rng.permutation(event_ids)

df_shuffled = df.loc[shuffled_event_ids]

assert len(df) == len(df_shuffled), "Shuffeling failed"

df = df_shuffled

# %%
# Fix the event ids
df.reset_index(inplace=True)

df.eval("evt = event_id", inplace=True)

event_ids = df["event_id"].unique()
new_event_ids = np.arange(len(event_ids))
mapping = {old:new for old, new in zip(event_ids, new_event_ids)}
mapping

N_events_before_reindexing = len(df["event_id"].unique())

df["event_id"] = df["event_id"].map(mapping)
# df_reindexed["event_id"] = df["event_id"].map(lambda x : new_event_ids[event_ids == x][0])

N_events_after_reindexing = len(df["event_id"].unique())

df.set_index(["event_id", "track_id"], inplace=True)

assert N_events_before_reindexing == N_events_after_reindexing, f"Reindexing lead to a loss of events. (before: #{N_events_before_reindexing}, after: {N_events_after_reindexing})"

# %%
# Generate new features

metric = np.diag(np.array([-1, -1, -1, 1])) # same as ROOT

def dot4v(v1, v2):
    '''

    Perform n four-vector dot products on vectors of shape (n, 4)
    (not a matrix product!)

    '''

    return np.einsum('ij,ij->i', v1, np.dot(v2, metric))

def constructVariables(df):

    df['diff_z'] = df['Tr_T_TrFIRSTHITZ'] - df['B_OWNPV_Z']

    # Construct B 'four-vectors'
    b4v = df[['B_PX', 'B_PY', 'B_PZ', 'B_PE']].values

    proj_array = []

    # For each event, get the arrays of track momenta
    for i, (_, (px, py, pz, pe)) in enumerate(df[['Tr_T_PX', 'Tr_T_PY', 'Tr_T_PZ', 'Tr_T_E']].iterrows()):

        # Construct an array 'four-vectors' of track mometa for this event (shape (nTracks, 4))
        t4v = np.vstack((px, py, pz, pe)).T

        # Broadcast the B four-vector to the same shape as the tracks, so they can be multiplied
        b = np.broadcast_to(b4v[i], (len(t4v), 4))

        # Perform the dot product on the vector of length nTracks
        proj = dot4v(b, t4v)

        proj_array.append(proj)

    df['P_proj'] = np.array(proj_array)

    # df['diff_pt'] = [ (tracks_pt.max() - tracks_pt) for tracks_pt in df['Tr_T_PT'] ]

    df['cos_diff_phi'] = np.array(list(map(lambda x : np.cos(x), df['B_LOKI_PHI'] - df['Tr_T_Phi'])))

    df['diff_eta'] = df['B_LOKI_ETA'] - df['Tr_T_Eta']

    df["SS"] = (df["Tr_ORIG_FLAGS"] == 1).astype(int)

    return df

df = constructVariables(df)

# %%
# Save the dataframe to a root file
with uproot.recreate(output_file) as file:
    print("Writing output file...")
    file["DecayTree"] = df.drop(columns="decay").reset_index()
    print("Done writing")

# %%








# df.reset_index(inplace=True)
# event_ids = df["event_id"].unique()
# new_event_ids = np.arange(len(event_ids))

# %%
#N_events_before_reindexing = len(df.index.unique("event_id"))

# df.set_index(["event_id", "track_id"], inplace=True)
#event_ids = df.index.unique("event_id")
#new_event_ids = np.arange(len(event_ids))
#df_reindexed = df.reindex(new_event_ids, level="event_id")
#
#N_events_after_reindexing = len(df_reindexed.index.unique("event_id"))
#
#if N_events_before_reindexing == N_events_after_reindexing:
#    print("Reindexing successful")
#else:
#    raise RuntimeError(f"Reindexing lead to a loss of events. (before: #{N_events_before_reindexing}, after: {N_events_after_reindexing})")

# %%