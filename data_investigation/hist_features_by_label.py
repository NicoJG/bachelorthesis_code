# %%
# Imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# Imports from this project
sys.path.insert(0,'..')
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning, get_hist, calc_pull

# %%
# Constant variables
output_dir = paths.plots_dir
output_dir.mkdir(parents=True, exist_ok=True)

label_keys = ["Tr_is_SS", "B_is_strange"]
label_names = {"Tr_is_SS": "Track membership", "B_is_strange": "Signal meson flavour"}


# %%
# Read in the feature keys
feature_keys = load_feature_keys(["extracted_mc", "direct_mc","extracted", "direct"])

# Read in the feature properties
feature_props = load_feature_properties()

# %%
# Read the input data
print("Read in the data...")
df = load_preprocessed_data(N_entries_max=100000)
print("Done reading input")

# %%
# Histogram function
def hist_feature_by_label(df, fkey, fprops, lkey, lname, output_pdf, allow_logx=True):
    
    is_feature_int_only = fprops[fkey]["int_only"]
    is_label_binary = len(fprops[lkey]["category_values"]) == 2
    is_feature_categorical = fprops[fkey]["feature_type"] == "categorical"
    is_feature_numerical = fprops[fkey]["feature_type"] == "numerical"
    assert is_feature_categorical or is_feature_numerical, f"The feature {fkey} is not categorical and not numerical..."
    
    lvalues = fprops[lkey]["category_values"]
    lvalue_names = fprops[lkey]["category_names"]
    
    if is_feature_categorical:
        fvalues = fprops[fkey]["category_values"]
        fvalue_names = fprops[fkey]["category_names"]
    
    # prepare the subplots
    if is_label_binary:
        # prepare the pull plot
        fig = plt.figure(figsize=(10,7))

        ax0 = plt.subplot2grid((4,2), (0,0), rowspan=3, colspan=1)
        ax1 = plt.subplot2grid((4,2), (0,1), rowspan=3, colspan=1)
        axs = [ax0, ax1]

        ax0 = plt.subplot2grid((4,2), (3,0), rowspan=1, sharex=ax0)
        ax1 = plt.subplot2grid((4,2), (3,1), rowspan=1, sharex=ax1)
        axs_pull = [ax0, ax1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True)

    fig.suptitle(f"{fkey} by {lname} ({lkey})")
    
    # TODO: all below rework + histograms.py rework of find_good_binning (with feature_properties.py)
    
    #bin_edges, bin_centers, is_categorical, is_logx = find_good_binning(df[feature_key], n_bins_max=200, lower_quantil=0.0001, higher_quantil=0.9999, allow_logx=allow_logx)
    
    x = []
    sigma = []

    line_styles = ["solid","solid","dotted","dotted","dased","dashed"]
    colors = [f"C{i}" for i in range(10)]

    for lvalue, lvalue_name, ls, c in zip(lvalues, lvalue_names, line_styles, colors):
        x_normed, sigma_normed = get_hist(x=df.query(f"{lkey}=={lvalue}")[fkey], 
                                          bin_edges=bin_edges, 
                                          normed=True, 
                                          is_categorical=is_categorical,
                                          categorical_values=bin_centers)
        x.append(x_normed)
        sigma.append(sigma_normed)

        for ax in axs:
            if is_feature_categorical:
                ax.bar(bin_centers, x_normed, yerr=sigma_normed,
                       fill=False,
                       tick_label=bin_centers,
                       label=lvalue_name,
                       edgecolor=c,
                       ecolor=c,
                       linestyle=ls)
            else:
                ax.hist(bin_centers, weights=x_normed, bins=bin_edges, 
                        histtype="step", 
                        label=lvalue_name, 
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

            ax.set_xlabel(fkey)
        
        axs_pull[0].set_ylabel("Pull")
    else:
        for ax in axs:
            ax.set_xlabel(fkey)
    
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