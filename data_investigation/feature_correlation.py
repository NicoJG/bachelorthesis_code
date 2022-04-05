# %%
# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from tqdm.auto import tqdm
import json

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning
from utils.merge_pdfs import merge_pdfs
from utils import paths

# %%
# Constants
output_dir = paths.plots_dir/"feature_correlation"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = paths.plots_dir/"feature_correlation.pdf"

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["extracted","direct"])

# Read in the feature properties
fprops = load_feature_properties()

# %%
# Read in the data
print("Read in the data...")
df_data = load_preprocessed_data(N_entries_max=1000000000)
print("Done reading input")

# %%
# Apply cuts to the data
lower_quantile = 0.0001
higher_quantile = 0.9999

cut_loss = {}

mask = True
for feature in tqdm(feature_keys, desc="Apply Feature Cuts"):
    if fprops[feature]["feature_type"] == "numerical":
        temp_mask = fprops[feature][f"quantile_{lower_quantile}"] <= df_data[feature]
        temp_mask &= df_data[feature] <= fprops[feature][f"quantile_{higher_quantile}"]
        
        # include the error value because else, to much tracks get lost
        if "error_value" in fprops[feature].keys():
            temp_mask |= df_data[feature] == fprops[feature]["error_value"]
        
        cut_loss[feature] = {}
        cut_loss[feature]["relative_loss"] = (~temp_mask).sum()/len(temp_mask)
        cut_loss[feature]["absolute_loss"] = (~temp_mask).sum()
        
        mask &= temp_mask
        
df_data_cut = df_data[mask]

# save the list of loss in tracks
df_cut_loss = pd.DataFrame.from_dict(cut_loss, orient="index")
df_cut_loss.rename_axis(index="feature", inplace=True)
df_cut_loss.sort_values(by="absolute_loss", ascending=False, inplace=True)
df_cut_loss.to_csv(output_dir/"00_feature_cut_loss.csv", float_format="%.6f")

print(f"Tracks before the cuts: {df_data.shape[0]}")
print(f"Tracks after the cuts: {df_data_cut.shape[0]}")
        
# %%
# Rearrange the features so that categorical features come last
numerical_features = [feature for feature in feature_keys if fprops[feature]["feature_type"] == "numerical"]
categorical_features = [feature for feature in feature_keys if fprops[feature]["feature_type"] == "categorical"]

feature_keys = numerical_features + categorical_features

# %% 
# Calculate the correlation matrix
df_corr = df_data_cut[feature_keys].corr()

# Save the correlation matrix
df_corr.to_csv(output_dir/"01_feature_corr_matrix.csv", float_format="%.6f")

# Plot the correlation matrix
N_features = len(feature_keys)
plt.figure(figsize=(N_features/4,N_features/4))
plt.title(f"""Feature Correlation
Evaluated with cuts at quantiles {lower_quantile}-{higher_quantile}
Tracks: {df_data_cut.shape[0]} / {df_data.shape[0]}""")

ax_img = plt.matshow(df_corr, vmin=-1, vmax=+1, fignum=0, cmap="seismic")

plt.xticks(ticks=range(N_features), labels=feature_keys, rotation=90)
plt.yticks(ticks=range(N_features), labels=feature_keys)

# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ax_img, cax=cax)

plt.tight_layout()
plt.savefig(output_dir/"01_feature_correlation.pdf")
plt.show()

# %%
# List the highest correlations

print("Calculating the correlation matrix...")
df_highest_corr = df_corr.copy()
print("Done calculating the correlation matrix.")
# set lower triangular matrix (with diagonal) to nan
triu_mask = np.triu(np.ones(df_highest_corr.shape), k=1).astype(bool)
df_highest_corr.where(triu_mask, np.nan, inplace=True)
# set all abs values lower than 0.7 to nan
df_highest_corr.where(np.abs(df_highest_corr)>0.7, np.nan, inplace=True)
# list of all pairs that have a high correlation
df_highest_corr.dropna(how="all", inplace=True)
df_highest_corr = df_highest_corr.stack().reset_index()
# print and save the list
df_highest_corr.rename(columns={"level_0":"f0", "level_1":"f1", 0:"correlation"}, inplace=True)
df_highest_corr.sort_values(by="correlation", ascending=False, key=abs, inplace=True)
df_highest_corr = df_highest_corr[["correlation", "f0", "f1"]]
df_highest_corr.reset_index(drop=True, inplace=True)
print(df_highest_corr)
df_highest_corr.to_csv(output_dir/"01_feature_correlation.csv", float_format="%.4f", index=False)

# %%
# Plot the feature pairs with the highest correlation (scatterplot and hist2d plot)

for i, (corr, f0, f1) in tqdm(df_highest_corr.iterrows(), total=df_highest_corr.shape[0], desc="Pair Plots"):
    
    
    # find binning for both features and both numerical and categorical feature_types
    bin_edges, bin_centers, is_logx = [], [], []
    for f in [f0,f1]:
        if fprops[f]["feature_type"] == "numerical":
            bin_res = find_good_binning(fprops[f], 
                                        n_bins_max=200, 
                                        lower_quantile=0.0001, 
                                        higher_quantile=0.9999, 
                                        allow_logx=True)
            bin_edges.append(bin_res[0])
            bin_centers.append(bin_res[1])
            is_logx.append(bin_res[2])
        elif fprops[f]["feature_type"] == "categorical":
            is_logx.append(False)
            fvals = fprops[f]["category_values"]
            fvals = np.array(fvals)
            bin_centers.append(fvals)
            # the bin edges are the centers +- 0.49 (for neighbouring values not 0.5)
            fedges = np.column_stack([fvals-0.49,fvals+0.49]).flatten()
            bin_edges.append(fedges)
            
    
    data0 = df_data[f0].to_numpy()
    data1 = df_data[f1].to_numpy()
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
    fig.suptitle(f"Pair Plot ('{f0}' vs '{f1}')\ncorrelation: {corr:.5f}")
    
    ax0.set_title("Scatter Plot")
    ax0.scatter(data0, data1, s=1.0, rasterized=True)
    ax0.set_xlabel(f0)
    ax0.set_ylabel(f1)
    
    ax1.set_title("2D Histogram")
    hist = ax1.hist2d(data0, data1, 
               bins=[bin_edges[0], bin_edges[1]], 
               density=False,
               norm=mpl.colors.LogNorm(),
               cmap="inferno",
               rasterized=True)
    ax1.set_xlabel(f0)
    ax1.set_ylabel(f1)
    ax1.set_facecolor('black')
    
    if is_logx[0]:
        ax1.set_xscale("log")
    if is_logx[1]:
        ax1.set_yscale("log")
        
    cbar = plt.colorbar(hist[3])
    cbar.ax.set_ylabel("Counts")
    
    plt.tight_layout()
    plt.savefig(output_dir/f"02_pair_plot_{i:02d}_{f0}_{f1}.pdf")
    #plt.show()
    plt.close()
    
# %%
# Merge all plots
merge_pdfs(output_dir, output_file)

# %%
