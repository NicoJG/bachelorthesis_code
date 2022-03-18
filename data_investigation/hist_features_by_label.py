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
from utils.histograms import find_good_binning, get_hist, calc_pull

# %%
# Constant variables

input_file = Path("/ceph/users/nguth/data/preprocesses_mc_Sim9b.root")

output_dir = Path("../plots")

output_dir.mkdir(parents=True, exist_ok=True)

label_keys = ["Tr_is_SS", "B_is_strange"]
label_values = {"Tr_is_SS": [0, 1], "B_is_strange": [0,1]}
label_value_names = {"Tr_is_SS": ["other", "SS"], "B_is_strange": ["Bd", "Bs"]}
label_names = {"Tr_is_SS": "Track membership", "B_is_strange": "Signal meson flavour"}

N_tracks = 1000000

load_batch_size = 10000

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)


# %%
# Read in the feature keys
with open("../features.json") as features_file:
    features_dict = json.load(features_file)
    
feature_keys = []
for k in ["extracted", "direct"]:
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
# Histogram function
def hist_feature_by_label(df, feature_key, label_key, label_values, label_value_names, label_name, output_pdf):
    # plot a pull plot if the label is binary
    is_binary = len(label_values) == 2
    if is_binary:
        fig = plt.figure(figsize=(10,7))

        ax0 = plt.subplot2grid((4,2), (0,0), rowspan=3, colspan=1)
        ax1 = plt.subplot2grid((4,2), (0,1), rowspan=3, colspan=1)
        axs = [ax0, ax1]

        ax0 = plt.subplot2grid((4,2), (3,0), rowspan=1, sharex=ax0)
        ax1 = plt.subplot2grid((4,2), (3,1), rowspan=1, sharex=ax1)
        axs_pull = [ax0, ax1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True)

    fig.suptitle(f"{feature_key} by {label_name}")
    
    bin_edges, bin_centers = find_good_binning(df[feature_key], n_bins_max=100)
    
    x = []
    sigma = []

    line_styles = ["solid","solid","dotted","dotted","dased","dashed"]

    for i, (l_val, l_val_name) in enumerate(zip(label_values, label_value_names)):
        x_normed, sigma_normed = get_hist(df.query(f"{label_key}=={l_val}")[feature_key], bin_edges, normed=True)

        for ax in axs:
            ax.hist(bin_centers, weights=x_normed, bins=bin_edges, 
                    histtype="step", 
                    label=l_val_name, 
                    color=f"C{i}",
                    linestyle=line_styles[i])
            ax.errorbar(bin_centers, x_normed, yerr=sigma_normed, 
                        fmt="none", 
                        color=f"C{i}")
            ax.legend(loc="best")

        x.append(x_normed)
        sigma.append(sigma_normed)

    axs[1].set_yscale("log")
    axs[0].set_ylabel("Frequency")
    axs[1].legend(loc="best")

    if is_binary:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])

        for ax in axs_pull:
            ax.hist(bin_centers, weights=pull, bins=bin_edges)
            ax.set_xlabel(feature_key)
        
        axs_pull[0].set_ylabel("Pull")
    else:
        for ax in axs:
            ax.set_xlabel(feature_key)
    
    fig.tight_layout()

    assert isinstance(output_pdf, PdfPages), "output_pdf must be matplotlib.backends.backend_pdf.PdfPages"
    output_pdf.savefig(fig)
    
    plt.close()
        

# %%
# Hist of all features by all labels

for label_key in label_keys:
    output_file = output_dir / f"features_by_{label_key}.pdf"

    output_pdf = PdfPages(output_file)

    for feature_key in tqdm(feature_keys, f"Features by {label_key}"):
        hist_feature_by_label(df, feature_key, label_key, 
                              label_values[label_key], 
                              label_value_names[label_key], 
                              label_names[label_key],
                              output_pdf)
    
    output_pdf.close()

# %%
