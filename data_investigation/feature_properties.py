# Properties that should be evaluated for each feature:
# mc_only: if the feature is only present in simulated data
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
man_feature_type = {"Tr_T_NbNonIsoTr_MinBDT_ult":"numerical",
                    "Tr_T_NbTrNonIso_sigtr":"numerical"}
man_error_value = {"Tr_T_Sum_of_trackp":None,
                   "Tr_T_Sum_of_trackpt":None,
                   "Tr_T_ConIso_p_ult":None,
                   "Tr_T_ConIso_pt_ult":None,
                   "Tr_T_Cone_asym_P":None,
                   "Tr_T_Cone_asym_Pt":None}
man_logx = {"Tr_T_Best_PAIR_D":True,
            "Tr_T_Best_PAIR_M_fromiso":True,
            "Tr_T_Best_PAIR_VCHI2":True,
            "Tr_T_PROBNNpi":False}

# %%
# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import paths
from utils.input_output import load_feature_keys, load_features_dict, load_feature_properties, load_preprocessed_data

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["extracted_mc", "direct_mc","extracted", "direct"])

# %%
# Read the input data
print("Read in the data...")
df_data = load_preprocessed_data(N_entries_max=10000000000)
print("Done reading input")

# %%
# Prepare the feature properties dictionary
if paths.feature_properties_file.is_file():
    fprops = load_feature_properties()
else:
    fprops = {feature:dict() for feature in feature_keys}

# %%
# Check if the feature is only present in MC simulated data
for feature in feature_keys:
    features_dict = load_features_dict()

    if feature in features_dict["extracted_mc"] or feature in features_dict["direct_mc"]:
        fprops[feature]["mc_only"] = True
    else:
        fprops[feature]["mc_only"] = False

# %%
# Feature Type
for feature in feature_keys:
    # Check for only integer values
    int_only = (df_data[feature] % 1 == 0).all()
    fprops[feature]["int_only"] = int_only
    
    # Choose if the feature could be categorical
    feature_type = "numerical"
    max_potential_categories = 20
    if int_only:
        unique_values = df_data[feature].unique()
        if len(df_data[feature].unique()) <= max_potential_categories:
            feature_type = "categorical"
    
    if feature in man_feature_type.keys():
        feature_type = man_feature_type[feature]
    
    fprops[feature]["feature_type"] = feature_type
    
# %%
# Handle categorical features
for feature in feature_keys:
    # insert dummy categories, so that the unique values are already in the file
    # the categories must be filled manually
    if fprops[feature]["feature_type"] != "categorical":
        continue
    
    uniq_vals = sorted(df_data[feature].astype(int).unique())
    
    if "category_values" not in fprops[feature].keys():
        fprops[feature]["category_values"] = uniq_vals
    else:
        if set(fprops[feature]["category_values"]) != set(uniq_vals):
            print(f"WARNING: There is an asymmetry in the category values found for '{feature}'")
            print(f"Categories found in the data: {uniq_vals}")
            print(f"Categories in feature_properties.json: {fprops[feature]['category_values']}")
    
    if "category_names" not in fprops[feature].keys():
        fprops[feature]["category_names"] = []
    

# %%
# Handle numerical features
for feature in feature_keys:
    if fprops[feature]["feature_type"] != "numerical":
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
        fprops[feature]["error_value"] = man_error_value[feature]
        if man_error_value[feature] is None:
            fprops[feature].pop("error_value")
    elif count_diff_mag >= count_diff_max_magnitude:
        # check for oders of magnitude
        error_val = uniq[0]
        # check if values are near the potential error value
        if not np.any((uniq[1:] > error_val-epsilon) & (uniq[1:] < error_val+epsilon)):
            fprops[feature]["error_value"] = error_val
            # for further analysis exclude the error values to NaN
            # WARNING: any analysis from this point on does not include the error value
            f_data = f_data[f_data != error_val]
            
    
    
    # calc the min and max values (for outliers)
    fprops[feature]["min"] = np.min(f_data)
    fprops[feature]["max"] = np.max(f_data)
    
    # calc the quantiles
    quantiles = [0.01,0.0001,0.5,0.99,0.9999]
    for q in quantiles:
        fprops[feature][f"quantile_{q}"] = np.quantile(f_data, q)
        
    # check if logx is sensible
    if "logx" not in fprops[feature].keys():
        fprops[feature]["logx"] = False
        
    min_magnitude_diff_for_logx = 4 # how many orders of magnitude should lie between the lower and higher end of values so that logx should be considered
    if np.all(f_data>0):
        mag_lower = np.log10(fprops[feature]["quantile_0.0001"])
        mag_higher = np.log10(fprops[feature]["quantile_0.9999"])
        mag_diff = mag_higher - mag_lower
        if mag_diff >= min_magnitude_diff_for_logx:
            fprops[feature]["logx"] = True
            
    if feature in man_logx:
        fprops[feature]["logx"] = man_logx[feature]
       
# %%
# Numpy JSON serialization
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# %%
# Write the feature properties to the json file
with open(paths.feature_properties_file, "w") as file:
    fprops_json = json.dumps(fprops, indent=2, cls=NpEncoder)
    # delete all line breaks in lists (between [ and ] )
    # https://stackoverflow.com/questions/71742728/regex-match-line-breaks-or-spaces-between-square-brackets
    fprops_json = re.sub(r"\[[^][]*]", lambda z: re.sub(r'\s+', '', z.group()), fprops_json)
    file.write(fprops_json)

# %%
# TODO: make functions
# TODO: make code more clean