# all relevant paths are inside ./utils/paths.py
from utils import paths
from datetime import datetime
from pathlib import Path

datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

ss_classifier_name = "SS_classifier"
paths.update_ss_classifier_name(ss_classifier_name)

B_classifier_name = f"B_classifier_{datetime_str}"
paths.update_B_classifier_name(B_classifier_name)

log_dir = Path("/ceph/users/nguth/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"snakemake_{datetime_str}.log"
log_pipe_command = f" |& tee -a {str(log_file)}"

rule master:
    input: str(paths.ss_classifier_eval_file), 
           str(paths.ss_classifier_feature_importance_file), 
           str(paths.B_classifier_eval_file),
           str(paths.B_classifier_feature_importance_file)

rule preprocess_training_data:
    input: str(paths.B2JpsiKstar_file), str(paths.Bs2DsPi_file)
    output: str(paths.preprocessed_data_file)
    threads: 50
    shell: f"python preprocess_training_data.py{log_pipe_command}"

rule train_ss_classifier:
    input: str(paths.preprocessed_data_file)
    output: str(paths.ss_classifier_model_file)
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1
    shell: f"python train_ss_classifier.py -n {ss_classifier_name}{log_pipe_command}"

rule eval_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_eval_file)
    threads: 50
    shell: f"python model_investigation/eval_ss_classifier.py -n {ss_classifier_name}{log_pipe_command}"

rule feature_importance_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_feature_importance_file)
    threads: 50
    shell: f"python model_investigation/feature_importance_ss_classifier.py -n {ss_classifier_name}{log_pipe_command}"

rule apply_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classified_data_file)
    threads: 50
    shell: f"python apply_ss_classifier.py -n {ss_classifier_name}{log_pipe_command}"

rule train_B_classifier:
    input: str(paths.ss_classified_data_file)
    output: str(paths.B_classifier_model_file)
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1 
    shell: f"python train_B_classifier.py -l -n {B_classifier_name}{log_pipe_command}"

rule eval_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_eval_file)
    threads: 50
    shell: f"python model_investigation/eval_B_classifier.py -n {B_classifier_name}{log_pipe_command}"

rule feature_importance_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_feature_importance_file)
    threads: 50
    shell: f"python model_investigation/feature_importance_B_classifier.py -n {B_classifier_name}{log_pipe_command}"
