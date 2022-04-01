# Properties that should be evaluated for each feature:
# feature_type: numerical or categorical
# int_only: if all values are only integers
# category_values: if categorical, the unique values representing categories
# category_names: if categorical, the names which the value at the same index represents
# units: if applicable the physical units of the feature
# error_value: if numerical, is there on value that is just arbitrarily chosen to represent something non numerical (e.g. an error)
# min,max: minimal and maximal values (to look for outliers)
# quantile_0.01, quantile_0.0001, quantile_0.99, quantile_0.9999: if numerical,
#   quantiles for plotting a good range (the error_value is excluded)
# logx: if the values are best represented using a log scale on the x axis
# description: short text describing the feature
# notes

# %%
# Manual corrections, because most algorithms here are just guesses and a human is better in estimating those properties:
man_feature_type = {}
man_error_value = {}
man_logx = {}

# %%
# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports from this project
sys.path.insert(0,'..')
from utils import paths
from utils.input_output import load_feature_keys, load_preprocessed_data

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["extracted", "direct"])

# %%
# Read the input data
print("Read in the data...")
df_data = load_preprocessed_data(N_entries_max=1000000,
                            batch_size=100000)
print("Done reading input")

# %%
# Prepare the feature properties DataFrame
if paths.feature_properties_file.is_file():
    df = pd.read_json(paths.feature_properties_file, orient="index")
else:
    df = pd.DataFrame({"feature":feature_keys})
    df.set_index("feature", inplace=True)

# %%
# Feature Type
for feature in feature_keys:
    # Check for only integer values
    int_only = (df_data[feature] % 1 == 0).all()
    df.loc[feature, "int_only"] = int_only
    
    # Choose if the feature could be categorical
    feature_type = "numerical"
    max_potential_categories = 20
    if int_only:
        unique_values = df_data[feature].unique()
        if len(df_data[feature].unique()) <= max_potential_categories:
            feature_type = "categorical"
    
    if feature in man_feature_type.keys():
        feature_type = man_feature_type[feature]
    
    df.loc[feature, "feature_type"] = feature_type
    
# %%
# Handle categorical features
for feature in feature_keys:
    # insert dummy categories, so that the unique values are already in the file
    # the categories must be filled manually
    if df.loc[feature, "feature_type"] != "categorical":
        continue
    
    if "category_values" not in df.columns:
        df["category_values"] = [None]*len(feature_keys)
    if "category_names" not in df.columns:
        df["category_names"] = [None]*len(feature_keys)
        
    if df.loc[feature, "category_values"] is None: 
        uniq_vals = sorted(df_data[feature].astype(int).unique())
        df.at[feature, "category_values"] = uniq_vals

# %%
# Handle numerical features
for feature in feature_keys:
    if df.loc[feature, "feature_type"] != "numerical":
        continue
    
    # make a copy of the feature data for easier use
    f_data = df_data[feature].to_numpy()
    
    # check for error values
    # error values are unique values with much more entries and no other entries near them
    epsilon = 0.5
    count_diff_max_magnitude = 3 # how many orders of magnitude should lie between the counts
    uniq, counts = np.unique(f_data,return_counts=True)
    sort_mask = np.argsort(-counts)
    uniq = uniq[sort_mask]
    counts = counts[sort_mask]
    count_diff_mag = np.log10(counts[0]) - np.log10(counts[1])
    if feature in man_error_value:
        df.loc[feature, "error_value"] = man_error_value[feature]
    elif count_diff_mag >= count_diff_max_magnitude:
        # check for oders of magnitude
        error_val = uniq[0]
        # check if values are near the potential error value
        if not np.any((uniq[1:] > error_val-epsilon) & (uniq[1:] < error_val+epsilon)):
            df.loc[feature, "error_value"] = error_val
            # for further analysis exclude the error values to NaN
            # WARNING: any analysis from this point on does not include the error value
            f_data = f_data[f_data != error_val]
            
    
    
    # calc the min and max values (for outliers)
    df.loc[feature, "min"] = np.min(f_data)
    df.loc[feature, "max"] = np.max(f_data)
    
    # calc the quantiles
    quantiles = [0.01,0.0001,0.5,0.99,0.9999]
    for q in quantiles:
        df.loc[feature, f"quantile_{q}"] = np.quantile(f_data, q)
        
    # check if logx is sensible
    if "logx" not in df.columns:
        df["logx"] = False
        
    min_magnitude_diff_for_logx = 3 # how many orders of magnitude should lie between the lower and higher end of values so that logx should be considered
    if np.all(f_data>0):
        mag_lower = np.log10(df.loc[feature, "quantile_0.01"])
        mag_higher = np.log10(df.loc[feature, "quantile_0.99"])
        mag_diff = mag_higher - mag_lower
        if mag_diff >= min_magnitude_diff_for_logx:
            df.loc[feature, "logx"] = True
            
    if feature in man_logx:
        df.loc[feature, "logx"] = man_logx[feature]
        
# %%
# Write the feature properties to the json file

# this is a workaround so that null values don't show up:
# https://stackoverflow.com/questions/30912746/pandas-remove-null-values-when-to-json
# temp_df = df.apply(lambda x: [x.dropna()], axis=1)

df.to_json(paths.feature_properties_file,
           orient="index",
           indent=2,
           double_precision=5)

# %%
# TODO: switch from df to dict
# TODO: make functions
# TODO: make code more clean
# TODO: actually check the properties