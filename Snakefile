# all relevant paths are inside ./utils/paths.py
from utils import paths
from datetime import datetime
from pathlib import Path

datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

ss_classifier_name = "SS_classifier"
paths.update_ss_classifier_name(ss_classifier_name)

B_classifier_name = f"B_classifier_{datetime_str}"
paths.update_B_classifier_name(B_classifier_name)

def log_pipe_command(rule_name):
    print(rule_name)
    log_dir = Path(f"/ceph/users/nguth/logs/snakemake_{datetime_str}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{rule_name}.log"
    return f"&>> {str(log_file)}"


rule master:
    input: str(paths.ss_classifier_eval_file), 
           str(paths.ss_classifier_feature_importance_file), 
           str(paths.B_classifier_eval_file),
           str(paths.B_classifier_feature_importance_file)

rule preprocess_training_data:
    input: str(paths.B2JpsiKstar_file), str(paths.Bs2DsPi_file)
    output: str(paths.preprocessed_data_file)
    params: 
        suffix=log_pipe_command("preprocess_training_data")
    threads: 50
    shell: "python preprocess_training_data.py{params.suffix}"

rule train_ss_classifier:
    input: str(paths.preprocessed_data_file)
    output: str(paths.ss_classifier_model_file)
    params: 
        suffix=log_pipe_command("train_ss_classifier"),
        model_name=f"{ss_classifier_name}"
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1
    shell: "python train_ss_classifier.py -n {params.model_name} {params.suffix}"

rule eval_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_eval_file)
    params: 
        suffix=log_pipe_command("eval_ss_classifier"),
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/eval_ss_classifier.py -n {params.model_name} {params.suffix}"

rule feature_importance_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_feature_importance_file)
    params: 
        suffix=log_pipe_command("feature_importance_ss_classifier"),
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/feature_importance_ss_classifier.py -n {params.model_name} {params.suffix}"

rule apply_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classified_data_file)
    params: 
        suffix=log_pipe_command("apply_ss_classifier"),
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python apply_ss_classifier.py -n {params.model_name} {params.suffix}"

rule train_B_classifier:
    input: str(paths.ss_classified_data_file)
    output: str(paths.B_classifier_model_file)
    params: 
        suffix=log_pipe_command("train_B_classifier"),
        model_name=f"{B_classifier_name}"
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1 
    shell: "python train_B_classifier.py -l -n {params.model_name} {params.suffix}"

rule eval_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_eval_file)
    params: 
        suffix=log_pipe_command("eval_B_classifier"),
        model_name=f"{B_classifier_name}"
    threads: 50
    shell: "python model_investigation/eval_B_classifier.py -n {params.model_name} {params.suffix}"

rule feature_importance_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_feature_importance_file)
    params: 
        suffix=log_pipe_command("feature_importance_B_classifier"),
        model_name=f"{B_classifier_name}"
    threads: 50
    shell: "python model_investigation/feature_importance_B_classifier.py -n {params.model_name} {params.suffix}"
