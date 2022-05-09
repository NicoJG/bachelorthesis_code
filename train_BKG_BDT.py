# %%
# Imports
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import json


# Local Imports
from utils.input_output import load_data_from_root, load_feature_keys
from utils import paths

# %%
# Constants

mc_file = paths.B2JpsiKS_MC_file
data_file = paths.B2JpsiKS_Data_file

mc_tree_key = "inclusive_Jpsi/DecayTree"
data_tree_key = "Bd2JpsiKSDetached/DecayTree"

# %%
# Load all relevant feature keys
bdt_features_mc = load_feature_keys(["features_BKG_BDT_mc","features_Lambda_cut"], file_path=paths.features_data_testing_file)
bdt_features_data = load_feature_keys(["features_BKG_BDT_data","features_Lambda_cut"], file_path=paths.features_data_testing_file)


# %%
# Find keys in the Trees
with uproot.open(mc_file)[mc_tree_key] as tree:
    mc_all_feature_keys = tree.keys()
    
with uproot.open(data_file)[data_tree_key] as tree:
    data_all_feature_keys = tree.keys()
    
with open(paths.internal_base_dir/"temp"/"mc_keys.json", "w") as file:
    json.dump({"keys":mc_all_feature_keys}, file, indent=2)
    
with open(paths.internal_base_dir/"temp"/"data_keys.json", "w") as file:
    json.dump({"keys":data_all_feature_keys}, file, indent=2)

# %%
# Load the data for the BDT
df_mc = load_data_from_root(mc_file, mc_tree_key, 
                            features=bdt_features_mc, 
                            cut="B0_BKGCAT==0",
                            n_threads=20,
                            N_entries_max=1000000000)
df_data = load_data_from_root(data_file, data_tree_key, 
                              features=bdt_features_data, 
                              #cut="B_M>5450", 
                              n_threads=20,
                              N_entries_max=1000000000)

# %%
# Rename the MC columns to fit the data
df_mc.rename(columns={mc:data for mc,data in zip(bdt_features_mc, bdt_features_data)}, inplace=True)

assert set(df_data.columns) == set(df_mc.columns)

# %%
# Create the label key
label_key = "B_is_signal"
df_data[label_key] = 0
df_mc[label_key] = 1

# %%
# Merge mc and data
df = pd.concat([df_data, df_mc])

# %%
# Prepare the cut of the Lambda Background
m_pi = 139.57039 # MeV
m_p = 938.272081 # MeV

def weird_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    return ( (E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2 )**0.5

E_piplus = (df["piplus_P"]**2 + m_pi**2)**0.5
E_pplus = (df["piplus_P"]**2 + m_p**2)**0.5
E_piminus = (df["piminus_P"]**2 + m_pi**2)**0.5
E_pminus = (df["piminus_P"]**2 + m_p**2)**0.5

df["m_pi-p+"] = weird_mass(E_piminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_pplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])
df["m_p-pi+"] = weird_mass(E_pminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_piplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])

# %%
# Make a veto feature
veto_m_min, veto_m_max = 1100., 1130. 

df["lambda_veto"] = (veto_m_min < df["m_pi-p+"]) & (df["m_pi-p+"] < veto_m_max) & (df["piplus_ProbNNp"] < 0.8)
df["lambda_veto"] |= (veto_m_min < df["m_p-pi+"]) & (df["m_p-pi+"] < veto_m_max) & (df["piminus_ProbNNp"] < 0.8)

# %%
# Plot the invariant masses for the cut of the Lambda Background
n_bins = 200
x_min, x_max = 1060.0 , 1180.0
bins = np.linspace(x_min, x_max, n_bins+1)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
ax0.hist(df["m_pi-p+"], bins=bins, alpha=0.5, label="m_pi-p+")
ax0.hist(df.query("lambda_veto==0")["m_pi-p+"], bins=bins, alpha=0.5, label="m_pi-p+ veto")
ax1.hist(df["m_p-pi+"], bins=bins, alpha=0.5, label="m_pi+p-")
ax1.hist(df.query("lambda_veto==0")["m_p-pi+"], bins=bins, alpha=0.5, label="m_pi+p- veto")

ax0.legend()
ax1.legend()
plt.show()




# %%
