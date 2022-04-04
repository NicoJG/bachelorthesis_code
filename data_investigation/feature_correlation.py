# %%
# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils import paths

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
df_data_cut = df_data
for feature in tqdm(feature_keys, desc="Apply Feature Cuts"):
    if fprops[feature]["feature_type"] == "numerical":
        mask = fprops[feature]["quantile_0.0001"] <= df_data_cut[feature]
        mask &= df_data_cut[feature] <= fprops[feature]["quantile_0.9999"]
        df_data_cut = df_data_cut[mask]

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

# Plot the correlation matrix
N_features = len(feature_keys)
plt.figure(figsize=(N_features/4,N_features/4))
plt.title("Feature Correlation")

ax_img = plt.matshow(df_corr, vmin=-1, vmax=+1, fignum=0, cmap="seismic")

plt.xticks(ticks=range(N_features), labels=feature_keys, rotation=90)
plt.yticks(ticks=range(N_features), labels=feature_keys)

# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ax_img, cax=cax)

plt.tight_layout()
plt.savefig(paths.plots_dir/"feature_correlation.pdf")
plt.show()

# %%
# List the highest correlations

temp_df = df_corr.copy()
# set lower triangular matrix (with diagonal) to nan
triu_mask = np.triu(np.ones(temp_df.shape), k=1).astype(bool)
temp_df.where(triu_mask, np.nan, inplace=True)
# set all abs values lower than 0.8 to nan
temp_df.where(np.abs(temp_df)>0.8, np.nan, inplace=True)
# list of all pairs that have a high correlation
temp_df.dropna(how="all", inplace=True)
temp_df = temp_df.stack().reset_index()
# print and save the list
temp_df.rename(columns={"level_0":"f0", "level_1":"f1", 0:"correlation"}, inplace=True)
temp_df.sort_values(by="correlation", ascending=False, inplace=True)
temp_df = temp_df[["correlation", "f0", "f1"]]
print(temp_df)
temp_df.to_csv(paths.plots_dir/"feature_correlation.csv", float_format="%.4f", index=False)

# %%
# TODO make scatter plots (or heatmap plots) for the highest correlation pairs