# all relevant paths are inside ./utils/paths.py
from utils import paths
from datetime import datetime
from pathlib import Path
import json

ss_classifier_name = "SS_classifier"
paths.update_ss_classifier_name(ss_classifier_name)

B_classifier_name = "B_classifier"
paths.update_B_classifier_name(B_classifier_name)

with open("features_B_classifier.json", "r") as file:
    B_classifier_names = list(map(lambda x: x.replace("features_",""), json.load(file).keys()))[:1]

B_classifier_names = [n for n in B_classifier_names if not "_with_" in n]

rule master:
    input: str(paths.ss_classifier_eval_file), 
           str(paths.ss_classifier_feature_importance_file), 
           expand(str(paths.models_dir / "{model_name}" / paths.B_classifier_eval_file.name), model_name=B_classifier_names),
           expand(str(paths.models_dir / "{model_name}" / paths.B_classifier_feature_importance_file.name), model_name=B_classifier_names)

localrules: eval_ss_classifier,
            feature_importance_ss_classifier, 
            apply_ss_classifier

rule preprocess_training_data:
    input: str(paths.B2JpsiKstar_file), str(paths.Bs2DsPi_file)
    output: str(paths.preprocessed_data_file)
    log: str(paths.logs_dir / "preprocess_training_data.log")
    threads: 50
    shell: "python preprocess_training_data.py &> {log}"

rule train_ss_classifier:
    input: str(paths.preprocessed_data_file)
    output: str(paths.ss_classifier_model_file)
    log: str(paths.logs_dir / "train_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python train_ss_classifier.py -n {params.model_name} &> {log}"

rule eval_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_eval_file)
    log: str(paths.logs_dir / "eval_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/eval_ss_classifier.py -n {params.model_name} &> {log}"

rule feature_importance_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classifier_feature_importance_file)
    log: str(paths.logs_dir / "feature_importance_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python model_investigation/feature_importance_ss_classifier.py -n {params.model_name} &> {log}"

rule apply_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classified_data_file)
    log: str(paths.logs_dir / "apply_ss_classifier.log")
    params: 
        model_name=f"{ss_classifier_name}"
    threads: 50
    shell: "python apply_ss_classifier.py -n {params.model_name} &> {log}"

rule train_B_classifier:
    input: str(paths.ss_classified_data_file)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    log: str(paths.logs_dir / "{model_name}" / "train_B_classifier.log")
    params: 
        model_name=f"{B_classifier_name}"
    threads: 50
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python train_B_classifier.py -t {threads} -l -n {wildcards.model_name} &> {log}"

rule eval_B_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_eval_file.name)
    log: str(paths.logs_dir / "{model_name}" / "eval_B_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python model_investigation/eval_B_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"

rule feature_importance_B_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_feature_importance_file.name)
    log: str(paths.logs_dir / "{model_name}" / "feature_importance_B_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python model_investigation/feature_importance_B_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"
