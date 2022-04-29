# all relevant paths are inside ./utils/paths.py
from utils import paths
from datetime import datetime
from pathlib import Path

datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

ss_classifier_name = "SS_classifier"
paths.update_ss_classifier_name(ss_classifier_name)

B_classifier_name = f"B_classifier_{datetime_str}"
paths.update_B_classifier_name(B_classifier_name)

log_dir = Path(f"/ceph/users/nguth/logs/snakemake_{datetime_str}")

rule master:
    input: str(paths.ss_classifier_eval_file), 
           str(paths.ss_classifier_feature_importance_file), 
           str(paths.B_classifier_eval_file),
           str(paths.B_classifier_feature_importance_file)

localrules: eval_ss_classifier,
            feature_importance_ss_classifier, 
            apply_ss_classifier,
            eval_B_classifier,  
            feature_importance_B_classifier

rule preprocess_training_data:
    input: str(paths.B2JpsiKstar_file), str(paths.Bs2DsPi_file)
    output: str(paths.preprocessed_data_file)
    log: str(log_dir/"preprocess_training_data.log")
    threads: 50
    shell: "python preprocess_training_data.py &> {log}"

rule train_ss_classifier:
    input: str(paths.preprocessed_data_file)
    output: str(paths.ss_classifier_model_file)
    log: str(log_dir/"train_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1
    shell: "python train_ss_classifier.py -n {params.model_name} &> {log}"

rule eval_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_eval_file)
    log: str(log_dir/"train_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/eval_ss_classifier.py -n {params.model_name} &> {log}"

rule feature_importance_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_feature_importance_file)
    log: str(log_dir/"feature_importance_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/feature_importance_ss_classifier.py -n {params.model_name} &> {log}"

rule apply_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classified_data_file)
    log: str(log_dir/"apply_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python apply_ss_classifier.py -n {params.model_name} &> {log}"

rule train_B_classifier:
    input: str(paths.ss_classified_data_file)
    output: str(paths.B_classifier_model_file)
    log: str(log_dir/"train_B_classifier.log")
    params: 
        model_name=f"{B_classifier_name}"
    threads: 1
    resources:
        MaxRunHours=4,
        request_memory=20*1024, # in MB
        request_gpus=0 
    shell: "python train_B_classifier.py -l -n {params.model_name} &> {log}"

rule eval_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_eval_file)
    log: str(log_dir/"eval_B_classifier.log")
    params: 
        model_name=f"{B_classifier_name}"
    threads: 50
    shell: "python model_investigation/eval_B_classifier.py -n {params.model_name} &> {log}"

rule feature_importance_B_classifier:
    input: str(paths.B_classifier_model_file)
    output: str(paths.B_classifier_feature_importance_file)
    log: str(log_dir/"feature_importance_B_classifier.log")
    params: 
        model_name=f"{B_classifier_name}"
    threads: 50
    shell: "python model_investigation/feature_importance_B_classifier.py -n {params.model_name} &> {log}"
