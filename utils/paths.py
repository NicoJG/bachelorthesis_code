from pathlib import Path

# Paths inside this project folder
internal_base_dir = Path(__file__).parent.parent.absolute()

features_file = internal_base_dir/"features.json"
features_SS_classifier_file = internal_base_dir/"features_SS_classifier.json"
features_B_classifier_file = internal_base_dir/"features_B_classifier.json"
feature_properties_file = internal_base_dir/"feature_properties.json"
plots_dir = internal_base_dir/"plots"

# Ceph Home
external_base_dir = Path("/ceph/users/nguth")

# Paths for the Snakefile
logs_dir = external_base_dir / "logs"

# Paths of data used by this project
B2JpsiKstar_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/B2JpsiKstar/Nov_2021_wgproduction/DTT_MC_Bd2JpsiKst_2016_26_Sim09b_DST.root")
Bs2DsPi_file = Path("/ceph/FlavourTagging/NTuples/ift/MC/Bs2DsPi/Mar_2022_wgproduction/DTT_MC_Bs2Dspi_2016_26_Sim09b_DST.root")

# Paths of data created and used by this project
data_dir = external_base_dir / "data"

preprocessed_data_file = data_dir / "preprocessed_mc_Sim9b.root"
ss_classified_data_file = data_dir / "SS_classified_mc_Sim9b.root"

# Paths to models trained by this project
models_dir = external_base_dir / "models"

# Paths to trained models for SS classification
def update_ss_classifier_name(model_name):
    global ss_classifier_dir, ss_classifier_model_file, ss_classifier_parameters_file, ss_classifier_train_test_split_file, ss_classifier_eval_dir, ss_classifier_eval_file, ss_classifier_feature_importance_dir, ss_classifier_feature_importance_file, ss_classifier_eval_data_file, ss_classifier_feature_importance_data_file
    
    assert isinstance(model_name, str), f"Please provide a str of what to add after '{str(models_dir)}/'!"
    
    ss_classifier_dir = models_dir / model_name
    ss_classifier_model_file = ss_classifier_dir/"model_ss_classifier.data"
    ss_classifier_parameters_file = ss_classifier_dir/"train_parameters.json"
    ss_classifier_train_test_split_file = ss_classifier_dir/"train_test_split.json"
    ss_classifier_eval_dir = ss_classifier_dir/"eval_plots"
    ss_classifier_eval_file = ss_classifier_dir/"eval_ss_classifier.pdf"
    ss_classifier_eval_data_file = ss_classifier_dir/"eval_results.json"
    ss_classifier_feature_importance_dir = ss_classifier_dir/"feature_importance"
    ss_classifier_feature_importance_file = ss_classifier_dir/"feature_importance_ss_classifier.pdf"
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


# Create directories that do not exist, that should exist always
for dir_ in [plots_dir, data_dir, models_dir]:
    if not dir_.is_dir():
        dir_.mkdir()
        print(f"Directory created: {str(dir_)}")
