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

# %%
# Constant variables

input_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")
output_file = Path("../plots/features_by_label.pdf")

output_file.parent.mkdir(parents=True, exist_ok=True)

labels = ["SS B_0", "SS B_s", "other B_0", "other B_s"]
label_ids = {l:i for i,l in enumerate(labels)}

N_tracks = 100000

load_batch_size = 10000

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)


# %%
# Read in the feature keys
with open("../features.json") as features_file:
    features_dict = json.load(features_file)
    
feature_keys = []
for k in ["extracted_mc", "direct_mc", "extracted", "direct"]:
    feature_keys.extend(features_dict[k])

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
# Histograms of all features by B_TRUEID
output_pdf = PdfPages(output_file)

for feature in tqdm(feature_keys, "Features"):
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(f"{feature} by B_TRUEID")
    B_IDs = [531,-531,511,-511]
    line_styles = ["solid", "solid", "dotted", "dotted",]
    for B_ID, line_style in zip(B_IDs, line_styles):
        bin_edges, bin_centers = find_good_binning(df[feature], n_bins_max=300)
        x_counts, _ = np.histogram(df[df["B_TRUEID"]==B_ID][feature], bins=bin_edges, density=True)
        for i in [0,1]: 
            axs[i].hist(bin_centers, weights=x_counts, bins=bin_edges, histtype="step", linestyle=line_style, label=f"B_TRUEID=={B_ID}")
    axs[1].set_yscale("log")
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    # plt.show()
    output_pdf.savefig(fig)
    plt.close()

output_pdf.close()

# %%

# Features to look out for B0 Bs classification:
# Tr_diff_pt
# Tr_T_SumBDT_ult
# Tr_T_ShareSamePVasSignal
# 