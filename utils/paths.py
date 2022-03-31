from pathlib import Path

# Paths inside this project
base_dir = Path(__file__).parent.parent.absolute()
features_file = base_dir/"features.json"
plots_dir = base_dir/"plots"

# Paths of data used by this project
B2JpsiKstar_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root")
Bs2DsPi_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root")

preprocessed_data_file = Path("/ceph/users/nguth/data/preprocessed_mc_Sim9b.root")