from pathlib import Path

# Paths inside this project folder
internal_base_dir = Path(__file__).parent.parent.absolute()

features_base_dir = internal_base_dir/"feature_lists"

features_file = features_base_dir/"features.json"
features_SS_classifier_file = features_base_dir/"features_SS_classifier.json"
features_B_classifier_file = features_base_dir/"features_B_classifier.json"
feature_properties_file = features_base_dir/"feature_properties.json"

plots_dir = internal_base_dir/"plots"

# Ceph Home
external_base_dir = Path("/ceph/users/nguth")

# Paths for the Snakefile
logs_dir = external_base_dir / "logs"

# Paths of data used by this project
B2JpsiKstar_files = [
    Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Mar_2022_wgproduction/DTT_MC_Bd2JpsiKst_2016_28r2_Sim10a_DST.root"),
    Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Mar_2022_wgproduction/DTT_MC_Bd2JpsiKst_2017_29r2_Sim10a_DST.root"),
    Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Mar_2022_wgproduction/DTT_MC_Bd2JpsiKst_2018_34_Sim10a_DST.root")
]
Bs2DsPi_files = [
    Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2016_28r2_Sim10a_DST.root"),
    Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2017_29r2_Sim10a_DST.root"),
    Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2018_34_Sim10a_DST.root")
]

B2JpsiKstar_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root")
Bs2DsPi_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root")

# Paths of data created and used by this project
data_dir = external_base_dir / "data"

preprocessed_data_file = data_dir / "preprocessed_mc.root"
ss_classified_data_file = data_dir / "SS_classified_mc.root"

# Paths to models trained by this project
models_dir = external_base_dir / "models"

# Paths to trained models for SS classification
def update_ss_classifier_name(model_name):
    global ss_classifier_dir, ss_classifier_model_file, ss_classifier_parameters_file, ss_classifier_train_test_split_file, ss_classifier_eval_dir, ss_classifier_eval_plots_file, ss_classifier_feature_importance_dir, ss_classifier_feature_importance_plots_file, ss_classifier_eval_data_file, ss_classifier_feature_importance_data_file
    
    assert isinstance(model_name, str), f"Please provide a str of what to add after '{str(models_dir)}/'!"
    
    ss_classifier_dir = models_dir / model_name
    ss_classifier_model_file = ss_classifier_dir/"model_ss_classifier.data"
    ss_classifier_parameters_file = ss_classifier_dir/"train_parameters.json"
    ss_classifier_train_test_split_file = ss_classifier_dir/"train_test_split.json"
    ss_classifier_eval_dir = ss_classifier_dir/"eval_plots"
    ss_classifier_eval_plots_file = ss_classifier_dir/"eval_ss_classifier.pdf"
    ss_classifier_eval_data_file = ss_classifier_dir/"eval_results.json"
    ss_classifier_feature_importance_dir = ss_classifier_dir/"feature_importance"
    ss_classifier_feature_importance_plots_file = ss_classifier_dir/"feature_importance_ss_classifier.pdf"
    ss_classifier_feature_importance_data_file = ss_classifier_feature_importance_dir/"feature_importance_ss_classifier.csv"
    
update_ss_classifier_name("SS_classifier")

# Paths to trained models for B classification
def update_B_classifier_name(model_name):
    global B_classifier_dir, B_classifier_model_file, B_classifier_parameters_file, B_classifier_train_test_split_file, B_classifier_eval_dir, B_classifier_eval_plots_file, B_classifier_feature_importance_dir, B_classifier_feature_importance_plots_file, B_classifier_eval_data_file, B_classifier_feature_importance_data_file
    
    assert isinstance(model_name, str), f"Please provide a str of what to add after '{str(models_dir)}/'!"
    
    B_classifier_dir = models_dir / model_name
    B_classifier_model_file = B_classifier_dir/"model_B_classifier.data"
    B_classifier_parameters_file = B_classifier_dir/"train_parameters.json"
    B_classifier_train_test_split_file = B_classifier_dir/"train_test_split.json"
    B_classifier_eval_dir = B_classifier_dir/"eval_plots"
    B_classifier_eval_plots_file = B_classifier_dir/"eval_B_classifier.pdf"
    B_classifier_eval_data_file = B_classifier_dir/"eval_results.json"
    B_classifier_feature_importance_dir = B_classifier_dir/"feature_importance"
    B_classifier_feature_importance_plots_file = B_classifier_dir/"feature_importance_B_classifier.pdf"
    B_classifier_feature_importance_data_file = B_classifier_feature_importance_dir/"feature_importance_B_classifier.csv"
    
update_B_classifier_name("B_classifier")


# Paths for the tests on data
B2JpsiKS_mc_dir = Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKS")
B2JpsiKS_data_dir = Path("/ceph/FlavourTagging/NTuples/ift/data/Nov_2021_wgproduction")

B2JpsiKS_mc_files = [B2JpsiKS_mc_dir/"DTT_MC_Bd2JpsiKS_2016_28r2_Sim09h_DST.root",
                     B2JpsiKS_mc_dir/"DTT_MC_Bd2JpsiKS_2017_29r2_Sim09h_DST.root",
                     B2JpsiKS_mc_dir/"DTT_MC_Bd2JpsiKS_2018_34_Sim09h_DST.root",]
B2JpsiKS_data_files = [B2JpsiKS_data_dir/"DTT_2016_Reco16Strip28r2_DIMUON_MagDown.root",
                       B2JpsiKS_data_dir/"DTT_2017_Reco17Strip29r2_DIMUON_MagDown.root",
                       B2JpsiKS_data_dir/"DTT_2018_Reco18Strip34_DIMUON_MagDown.root",
                       B2JpsiKS_data_dir/"DTT_2016_Reco16Strip28r2_DIMUON_MagUp.root",
                       B2JpsiKS_data_dir/"DTT_2017_Reco17Strip29r2_DIMUON_MagUp.root",
                       B2JpsiKS_data_dir/"DTT_2018_Reco18Strip34_DIMUON_MagUp.root"]

features_data_testing_file = features_base_dir/"features_data_testing.json"

data_testing_B_ProbBs_file = data_dir/"B2JpsiKS_B_ProbBs.root"

data_testing_plots_dir = plots_dir/"test_on_data"
data_testing_plots_file = plots_dir/"test_on_data.pdf"

# Paths to trained models for the BKG removal BDT
def update_bkg_bdt_name(model_name):
    global bkg_bdt_dir, bkg_bdt_model_file, bkg_bdt_parameters_file, bkg_bdt_train_test_split_file, bkg_bdt_eval_dir, bkg_bdt_eval_plots_file, bkg_bdt_eval_data_file
    
    assert isinstance(model_name, str), f"Please provide a str of what to add after '{str(models_dir)}/'!"
    
    bkg_bdt_dir = models_dir / model_name
    bkg_bdt_model_file = bkg_bdt_dir/"model_bkg_bdt.data"
    bkg_bdt_parameters_file = bkg_bdt_dir/"train_parameters.json"
    bkg_bdt_train_test_split_file = bkg_bdt_dir/"train_test_split.json"
    bkg_bdt_eval_dir = bkg_bdt_dir/"eval_plots"
    bkg_bdt_eval_plots_file = bkg_bdt_dir/"eval_bkg_bdt.pdf"
    bkg_bdt_eval_data_file = bkg_bdt_dir/"eval_results.json"
    
update_bkg_bdt_name("BKG_BDT")


# Create directories that do not exist, that should exist always
for dir_ in [plots_dir, data_dir, models_dir]:
    if not dir_.is_dir():
        dir_.mkdir()
        print(f"Directory created: {str(dir_)}")
