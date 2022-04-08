# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning, get_hist, calc_pull
from utils.merge_pdfs import merge_pdfs

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
df = load_preprocessed_data(N_entries_max=100000000)
print("Done reading input")

# %%
# Histogram functions
def prepare_subplots_grid(draw_pull, add_logx, add_logy, fkey):
    "Returns the figure and axes of what to plot on in a grid depending on whether to also plot additional plots with logarthmic axes"
    if not draw_pull:
        raise RuntimeError("Not drawing a pull plot is not implemented")
    
    axs = {}
    
    if not add_logx and not add_logy:
        # structure with gridding
        # normal (3) 
        # ------
        # pull   (1)
        # (1)
        fig = plt.figure(figsize=(5,7))

        axs["normal"] = plt.subplot2grid((4,1), (0,0), rowspan=3, colspan=1)

        axs["pull_normal"] = plt.subplot2grid((4,1), (3,0), rowspan=1, sharex=axs["normal"])
    
    elif add_logy and not add_logx:
        # structure with gridding
        # normal (3) 
        # --------
        # logy   (3)
        # -------
        # pull   (1)
        # (1)
        fig = plt.figure(figsize=(10,12))

        axs["normal"] = plt.subplot2grid((7,1), (0,0), rowspan=3, colspan=1)
        axs["logy"]   = plt.subplot2grid((7,1), (3,0), rowspan=3, colspan=1)

        axs["pull_normal"] = plt.subplot2grid((7,1), (6,0), rowspan=1, sharex=axs["normal"])
    elif add_logx and not add_logy:
        # structure with gridding
        # normal | logx (3) 
        # -------------
        # pull   | pull (1)
        # (1)    | (1)
        fig = plt.figure(figsize=(10,7))

        axs["normal"] = plt.subplot2grid((4,2), (0,0), rowspan=3, colspan=1)
        axs["logx"]   = plt.subplot2grid((4,2), (0,1), rowspan=3, colspan=1)

        axs["pull_normal"] = plt.subplot2grid((4,2), (3,0), rowspan=1, sharex=axs["normal"])
        axs["pull_logx"]   = plt.subplot2grid((4,2), (3,1), rowspan=1, sharex=axs["logx"])
    elif add_logx and add_logx:
        # structure with gridding
        # normal | logx          (3) 
        # -------------
        # logy   | logx and logy (3) 
        # -------------
        # pull   | pull          (1)
        # (1)    | (1)
        fig = plt.figure(figsize=(10,12))

        axs["normal"] = plt.subplot2grid((7,2), (0,0), rowspan=3, colspan=1)
        axs["logx"]   = plt.subplot2grid((7,2), (0,1), rowspan=3, colspan=1)
        
        axs["logy"]        = plt.subplot2grid((7,2), (3,0), rowspan=3, sharex=axs["normal"])
        axs["logx_logy"]   = plt.subplot2grid((7,2), (3,1), rowspan=3, sharex=axs["logx"])

        axs["pull_normal"] = plt.subplot2grid((7,2), (6,0), rowspan=1, sharex=axs["normal"])
        axs["pull_logx"]   = plt.subplot2grid((7,2), (6,1), rowspan=1, sharex=axs["logx"])
        
    # set up the x and y scales
    if "logx" in axs.keys():
        axs["logx"].set_xscale("log")
    if "logy" in axs.keys():
        axs["logy"].set_yscale("log")
    if "logx_logy" in axs.keys():
        axs["logx_logy"].set_xscale("log")
        axs["logx_logy"].set_yscale("log")
        
    if "pull_logx" in axs.keys():
        axs["pull_logx"].set_xscale("log")
    
    # set up the axis labels
    axs["normal"].set_ylabel("density")
    axs["pull_normal"].set_ylabel("pull")
    axs["pull_normal"].set_xlabel(fkey)
    
    if "pull_logx" in axs.keys():
        axs["pull_logx"].set_xlabel(fkey)
    
    return fig, axs


def hist_categorical_feature_by_label(ax, pull_ax, df, fkey, fprops, lkey, lvalues, lvalue_names):
    # save the histogrammed values for the pull plot
    x = []
    sigma = []
    
    # different colors and linestyles for each label value
    line_styles = ["solid","solid","dotted","dotted","dased","dashed"]
    colors = [f"C{i}" for i in range(10)]
    
    # get the binning (category values)
    fvalues = fprops["category_values"]
    fvalue_names = fprops["category_names"]
    
    # iterate through each label value
    for lvalue, lvalue_name, ls, c in zip(lvalues, lvalue_names, line_styles, colors):
        x_normed, sigma_normed = get_hist(x=df.query(f"{lkey}=={lvalue}")[fkey],
                                          is_categorical=True,
                                          categorical_values=fvalues,
                                          normed=True)
        
        x.append(x_normed)
        sigma.append(sigma_normed)
        
        if fvalue_names:
            tick_labels = [f"{fval}\n{fval_name}" for fval, fval_name in zip(fvalues, fvalue_names)]
        else:
            tick_labels = [f"{fval}" for fval in fvalues]
            
        # plot the bar plot
        ax.bar(tick_labels, x_normed, yerr=sigma_normed,
                          tick_label=tick_labels,
                          fill=False,
                          label=lvalue_name,
                          edgecolor=c,
                          ecolor=c,
                          linestyle=ls)
    
    # plot the pull plot
    if pull_ax is not None:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])
        pull_ax.bar(tick_labels, pull, tick_label=tick_labels)
    
    
def hist_numerical_feature_by_label(ax, pull_ax, is_logx, df, fkey, fprops, lkey, lvalues, lvalue_names):
    # save the histogrammed values for the pull plot
    x = []
    sigma = []

    # different colors and linestyles for each label value
    line_styles = ["solid","solid","dotted","dotted","dased","dashed"]
    colors = [f"C{i}" for i in range(10)]
    
    # calculate the binning
    bin_res = find_good_binning(fprops, 
                                n_bins_max=300, 
                                lower_quantile=0.0001,
                                higher_quantile=0.9999,
                                allow_logx=False,
                                force_logx=is_logx)
    bin_edges, bin_centers, _ = bin_res
        
    
    # iterate through each label value
    for lvalue, lvalue_name, ls, c in zip(lvalues, lvalue_names, line_styles, colors):
        x_normed, sigma_normed = get_hist(x=df.query(f"{lkey}=={lvalue}")[fkey], bin_edges=bin_edges, normed=True)
        
        x.append(x_normed)
        sigma.append(sigma_normed)
        
        # plot the hist with errorbars
        ax.hist(bin_centers, 
                           weights=x_normed, 
                           bins=bin_edges, 
                           histtype="step", 
                           label=lvalue_name, 
                           color=c,
                           linestyle=ls,
                           alpha=0.8)
        ax.errorbar(bin_centers, x_normed, 
                               yerr=sigma_normed, 
                               fmt="none", 
                               color=c,
                               alpha=0.8)
    
    # plot the pull plot
    if pull_ax is not None:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])
        
        pull_ax.hist(bin_centers, 
                     weights=pull, 
                     bins=bin_edges, 
                     histtype="stepfilled")
    

def hist_feature_by_label(df, fkey, fprops, lkey, lprops, output_file, add_logx=False, add_logy=False):
    
    assert lprops["feature_type"] == "categorical", f"The label ({lkey}) has to be categorical."
    
    assert fprops["feature_type"] in ["categorical", "numerical"], f"The feature ({fkey}) is not categorical and not numerical..."
    
    lvalues = lprops["category_values"]
    lvalue_names = lprops["category_names"]
    if not lvalue_names:
        lvalue_names = lvalues
        
    draw_pull = len(lvalues)==2
    
    # only add logarithmic x axis if the feature is float numerical and x_min>0
    add_logx = add_logx and fprops["feature_type"]=="numerical" and not fprops["int_only"] and fprops["quantile_0.0001"]>0.0
    
    # prepare the subplots
    fig, axs = prepare_subplots_grid(draw_pull, add_logx, add_logy, fkey)

    # set the figure title
    plot_title = f"{fkey} by {lkey}"
    if "error_value" in fprops.keys():
        error_val = fprops['error_value']
        error_val_counts = (df[fkey] == error_val).sum()
        error_val_proportion = error_val_counts / df.shape[0]
        plot_title += f"\nwithout the error value {error_val} (proportion: {error_val_proportion*100:.2f}%)"
    fig.suptitle(plot_title)
        
    if fprops["feature_type"] == "categorical":
        hist_categorical_feature_by_label(axs["normal"], axs["pull_normal"], df, fkey, fprops, lkey, lvalues, lvalue_names)
        if "logy" in axs.keys():
            hist_categorical_feature_by_label(axs["logy"], None, df, fkey, fprops, lkey, lvalues, lvalue_names)
    elif fprops["feature_type"] == "numerical":
        for ax_key in ["normal", "logx", "logy", "logx_logy"]:
            if ax_key in axs.keys():
                is_logx = "logx" in ax_key
                if f"pull_{ax_key}" in axs.keys():
                    pull_ax = axs[f"pull_{ax_key}"]
                else:
                    pull_ax = None
                hist_numerical_feature_by_label(axs[ax_key], pull_ax, is_logx, df, fkey, fprops, lkey, lvalues, lvalue_names)
    
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close()

# %%
# Hist of all features by all labels

for label_key in label_keys:
    output_label_dir = output_dir / f"features_by_{label_key}"
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    output_label_file = output_dir / f"features_by_{label_key}.pdf"

    for feature_key in tqdm(feature_keys, f"Features by {label_key}"):
        output_file = output_label_dir/f"{feature_key}.pdf"
        hist_feature_by_label(df, 
                              feature_key, feature_props[feature_key], 
                              label_key, feature_props[label_key],
                              output_file,
                              add_logx=True,
                              add_logy=True)
        
    merge_pdfs(output_label_dir, output_label_file)



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