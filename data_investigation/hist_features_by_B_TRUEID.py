# %%
# Imports
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import uproot
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Imports from this repository
sys.path.insert(0,'..')
from utils.histograms import find_good_binning
from utils.merge_pdfs import merge_pdfs

# %%
# Constant variables

input_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")
temp_dir = Path("../plots/features_by_B_TRUEID")
output_file = Path("../plots/features_by_B_TRUEID.pdf")

temp_dir.mkdir(parents=True, exist_ok=True)
output_file.parent.mkdir(parents=True, exist_ok=True)

N_tracks = 100000

load_batch_size = 10000

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)


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
# Read in the feature keys
with open("../features.json") as features_file:
    features_dict = json.load(features_file)
    
feature_keys = []
for k in ["extracted_mc", "direct_mc", "extracted", "direct"]:
    feature_keys.extend(features_dict[k])

# %%
# Histograms of all features by B_TRUEID
output_pdf = PdfPages(output_file)
for feature in tqdm(feature_keys, "Features by B_TRUEID"):
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(f"{feature} by B_TRUEID")
    B_IDs = [531,-531,511,-511]
    line_styles = ["solid", "solid", "dotted", "dotted",]
    for B_ID, line_style in zip(B_IDs, line_styles):
        bin_edges, _ = find_good_binning(df[feature])
        axs[0].hist(df[df["B_TRUEID"]==B_ID][feature], bins=bin_edges, histtype="step", linestyle=line_style, label=f"B_TRUEID=={B_ID}")
        axs[1].hist(df[df["B_TRUEID"]==B_ID][feature], bins=bin_edges, histtype="step", linestyle=line_style, label=f"B_TRUEID=={B_ID}")
    axs[1].set_yscale("log")
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    # plt.show()
    output_pdf.savefig(fig)
    plt.close()

output_pdf.close()

# %%
