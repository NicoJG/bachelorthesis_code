# %%
# Imports
import pathlib
import uproot
import pandas as pd
import numpy as np

# %%
# Constant variables
input_files = [
    "/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root",
    "/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root"]

output_file = "/ceph/users/nguth/data/preprocesses_mc_Sim9b.root"

N_events_per_dataset = 1000

# %%
# Read the data
with (uproot.open(input_files[0])["DecayTree"] as tree0,
     uproot.open(input_files[1])["DecayTree"] as tree1):
    trees = [tree0, tree1]
    print(trees[0].keys())

print(trees[0].keys())
# %%
