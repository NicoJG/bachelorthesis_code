from pathlib import Path

# Paths inside this project folder
base_dir = Path(__file__).parent.parent.absolute()
features_file = base_dir/"features.json"
feature_properties_file = base_dir/"feature_properties.json"
plots_dir = base_dir/"plots"

# Paths of data used by this project
B2JpsiKstar_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root")
Bs2DsPi_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root")

# Paths of data created and used by this project
data_dir = Path("/ceph/users/nguth/data")

preprocessed_data_file = data_dir/"preprocessed_mc_Sim9b.root"

# Paths to models trained by this project
models_dir = Path("/ceph/users/nguth/models")

# Paths to trained models for SS classification
def update_ss_classifier_dir(dir_name):
    global ss_classifier_dir, ss_classifier_model_file, ss_classifier_parameters_file, ss_classifier_train_test_split_file, ss_classifier_eval_dir, ss_classifier_eval_file
    
    assert isinstance(dir_name, str), f"Please provide a str of what to add after '{str(models_dir)}/'!"
    
    ss_classifier_dir = models_dir / dir_name
    ss_classifier_model_file = ss_classifier_dir/"model.data"
    ss_classifier_parameters_file = ss_classifier_dir/"train_parameters.json"
    ss_classifier_train_test_split_file = ss_classifier_dir/"train_test_split.json"
    ss_classifier_eval_dir = ss_classifier_dir/"eval_plots"
    ss_classifier_eval_file = ss_classifier_dir/"eval_ss_classifier.pdf"
    
update_ss_classifier_dir("SS_classifier")


# Create directories that do not exist, that should exist always
for dir_ in [plots_dir, data_dir, models_dir]:
    if not dir_.is_dir():
        dir_.mkdir()
        print(f"Directory created: {str(dir_)}")
