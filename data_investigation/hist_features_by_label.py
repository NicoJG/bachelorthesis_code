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

N_tracks_max = 1000000000

load_batch_size = 10000



# %%
# Read in the feature keys
with open("../features.json") as features_file:
    features_dict = json.load(features_file)
    
feature_keys = []
for k in ["extracted", "direct"]:
    feature_keys.extend(features_dict[k])

# %%
# Read number of tracks
with uproot.open(input_file)["DecayTree"] as tree:
    N_tracks_in_file = tree.num_entries

N_tracks = np.min([N_tracks_in_file, N_tracks_max])

print(f"Tracks in the dataset: {N_tracks_in_file}")
print(f"Tracks to use for plotting: {N_tracks}")

N_batches_estimate = np.ceil(N_tracks / load_batch_size).astype(int)
# %%
# Read the input data
print("Read in the data...")

df = pd.DataFrame()
with uproot.open(input_file)["DecayTree"] as tree:
    tree_iter = tree.iterate(entry_stop=N_tracks, step_size=load_batch_size, library="pd")
    for temp_df in tqdm(tree_iter, "Track batches", total=N_batches_estimate):
        temp_df.set_index("index", inplace=True)
        df = pd.concat([df, temp_df])

print("Done reading input")

# %%
# Histogram function
def hist_feature_by_label(df, feature_key, label_key, label_values, label_value_names, label_name, output_pdf, allow_logx=True):
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
    
    bin_edges, bin_centers, is_categorical, is_logx = find_good_binning(df[feature_key], n_bins_max=200, allow_logx=allow_logx)
    
    x = []
    sigma = []

    line_styles = ["solid","solid","dotted","dotted","dased","dashed"]
    colors = [f"C{i}" for i in range(10)]

    for l_val, l_val_name, ls, c in zip(label_values, label_value_names, line_styles, colors):
        x_normed, sigma_normed = get_hist(x=df.query(f"{label_key}=={l_val}")[feature_key], 
                                          bin_edges=bin_edges, 
                                          normed=True, 
                                          is_categorical=is_categorical,
                                          categorical_values=bin_centers)
        x.append(x_normed)
        sigma.append(sigma_normed)

        for ax in axs:
            if is_categorical:
                ax.bar(bin_centers, x_normed, yerr=sigma_normed,
                       fill=False,
                       tick_label=bin_centers,
                       label=l_val_name,
                       edgecolor=c,
                       ecolor=c,
                       linestyle=ls)
            else:
                ax.hist(bin_centers, weights=x_normed, bins=bin_edges, 
                        histtype="step", 
                        label=l_val_name, 
                        color=c,
                        linestyle=ls)
                ax.errorbar(bin_centers, x_normed, yerr=sigma_normed, 
                            fmt="none", 
                            color=c)

    axs[0].set_ylabel("Frequency")
    axs[0].legend(loc="best")

    axs[1].set_yscale("log")
    axs[1].legend(loc="best")

    if is_logx:
        axs[0].set_xscale("log")
        axs[1].set_xscale("log")

    if is_binary:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])

        for ax in axs_pull:
            if is_categorical:
                ax.bar(bin_centers, pull, tick_label=bin_centers)
            else:
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

    return is_logx

# %%
# Hist of all features by all labels

for label_key in label_keys:
    output_file = output_dir / f"features_by_{label_key}.pdf"

    output_pdf = PdfPages(output_file)

    for feature_key in tqdm(feature_keys, f"Features by {label_key}"):
        is_logx = hist_feature_by_label(df, feature_key, label_key, 
                                        label_values[label_key], 
                                        label_value_names[label_key], 
                                        label_names[label_key],
                                        output_pdf)
        if is_logx:
            hist_feature_by_label(df, feature_key, label_key, 
                                label_values[label_key], 
                                label_value_names[label_key], 
                                f"{label_names[label_key]} \n now without logx for comparison",
                                output_pdf,
                                allow_logx=False)

    output_pdf.close()

# %%

# strange Features:

# B flavour:
# Tr_diff_pt
# Tr_T_ACHI2DOCA
# Tr_T_ShareSamePVasSignal

# look weird:
# Tr_T_Best_PAIR_DCHI
# Tr_T_IPCHI2_trMother
# Tr_T_VeloCharge (zackig)
# Tr_T_IP_trMother


# SS vs other:
# Tr_T_VeloCharge
# Tr_T_AALLSAMEBP
# Tr_T_ShareSamePVasSignal
# Tr_T_SumMinBDT_ult

# Features with visual difference

# B flavour
# Tr_diff_pt
# Tr_diff_p 
# Tr_diff_eta
# Tr_T_ACHI2DOCA
# Tr_T_SumBDT_sigtr
# Tr_T_SumBDT_ult
# Tr_T_NbNonIsoTr_MinBDT_ult
# Tr_T_SumMinBDT_ult 

# SS vs other
# Tr_diff_z
# Tr_diff_eta 
# Tr_T_Sum_of_trackp
# Tr_T_Ntr_incone
# Tr_T_THETA
# Tr_T_Sum_of_trackpt
# Tr_T_SumBDT_ult
# Tr_T_Cone_asym_Pt
# Tr_T_Eta
# Tr_T_PT
# Tr_T_ConIso_pt_ult
# Tr_T_ConIso_p_ult