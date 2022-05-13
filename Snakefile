# all relevant paths are inside ./utils/paths.py
from utils import paths
from datetime import datetime
from pathlib import Path
import json

SS_classifier_name_main = "SS_classifier"
B_classifier_name_main = "B_classifier"
BKG_BDT_name_main = "BKG_BDT"

paths.update_ss_classifier_name(SS_classifier_name_main)
paths.update_B_classifier_name(B_classifier_name_main)
paths.update_bkg_bdt_name(BKG_BDT_name_main)

with open(paths.features_SS_classifier_file, "r") as file:
    SS_classifier_names = list(map(lambda x: x.replace("features_",""), json.load(file).keys()))[:]

with open(paths.features_B_classifier_file, "r") as file:
    B_classifier_names = list(map(lambda x: x.replace("features_",""), json.load(file).keys()))[:]

rule master:
    input: expand(str(paths.models_dir / "{model_name}" / paths.ss_classifier_eval_plots_file.name), model_name=SS_classifier_names),
           expand(str(paths.models_dir / "{model_name}" / paths.ss_classifier_feature_importance_plots_file.name), model_name=SS_classifier_names),
           expand(str(paths.models_dir / "{model_name}" / paths.B_classifier_eval_plots_file.name), model_name=B_classifier_names),
           expand(str(paths.models_dir / "{model_name}" / paths.B_classifier_feature_importance_plots_file.name), model_name=B_classifier_names),
           str(paths.bkg_bdt_eval_plots_file),
           str(paths.data_testing_B_ProbBs_file),
           str(paths.data_testing_plots_file)

rule preprocess_training_data:
    input: str(paths.B2JpsiKstar_file), str(paths.Bs2DsPi_file)
    output: str(paths.preprocessed_data_file)
    log: str(paths.logs_dir / "preprocess_training_data.log")
    threads: 20
    resources:
        MaxRunHours=4,
        request_memory=100*1024, # in MB
        request_gpus=0
    shell: "python preprocess_training_data.py -t {threads} &> {log}"

rule train_ss_classifier:
    input: str(paths.preprocessed_data_file)
    output: str(paths.models_dir / "{model_name}" / paths.ss_classifier_model_file.name)
    log: str(paths.logs_dir / "{model_name}" / "train_ss_classifier.log")
    threads: 40
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python train_ss_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"

rule eval_ss_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.ss_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.ss_classifier_eval_plots_file.name)
    log: str(paths.logs_dir / "{model_name}" / "eval_ss_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python model_investigation/eval_ss_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"

rule feature_importance_ss_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.ss_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.ss_classifier_feature_importance_plots_file.name)
    log: str(paths.logs_dir / "{model_name}" / "feature_importance_ss_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python model_investigation/feature_importance_ss_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"

rule apply_ss_classifier:
    input: str(paths.ss_classifier_model_file)
    output: str(paths.ss_classified_data_file)
    log: str(paths.logs_dir / "apply_ss_classifier.log")
    threads: 20
    params: 
        model_name=f"{SS_classifier_name_main}"
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python apply_ss_classifier.py -t {threads} -n {params.model_name} &> {log}"

rule train_B_classifier:
    input: str(paths.ss_classified_data_file)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    log: str(paths.logs_dir / "{model_name}" / "train_B_classifier.log")
    threads: 10
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1,
        #Requirements='(machine=="beagle.e5.physik.tu-dortmund.de")||(machine=="heemskerck.e5.physik.tu-dortmund.de")'
        Requirements='(machine=="tarek.e5.physik.tu-dortmund.de")'
    shell: "python train_B_classifier.py -g -t {threads} -l -n {wildcards.model_name} &> {log}"

rule eval_B_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_eval_plots_file.name)
    log: str(paths.logs_dir / "{model_name}" / "eval_B_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python model_investigation/eval_B_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"

rule feature_importance_B_classifier:
    input: str(paths.models_dir / "{model_name}" / paths.B_classifier_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.B_classifier_feature_importance_plots_file.name)
    log: str(paths.logs_dir / "{model_name}" / "feature_importance_B_classifier.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python model_investigation/feature_importance_B_classifier.py -t {threads} -n {wildcards.model_name} &> {log}"



# Tests on data
rule train_BKG_BDT:
    output: str(paths.models_dir / "{model_name}" / paths.bkg_bdt_model_file.name)
    log: str(paths.logs_dir / "{model_name}" / "train_BKG_BDT.log")
    threads: 40
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=0
    shell: "python train_BKG_BDT.py -t {threads} -n {wildcards.model_name} &> {log}"

rule eval_BKG_BDT:
    input: str(paths.models_dir / "{model_name}" / paths.bkg_bdt_model_file.name)
    output: str(paths.models_dir / "{model_name}" / paths.bkg_bdt_eval_plots_file.name)
    log: str(paths.logs_dir / "{model_name}" / "eval_BKG_BDT.log")
    threads: 20
    resources:
        MaxRunHours=1,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python model_investigation/eval_BKG_BDT.py -t {threads} -n {wildcards.model_name} &> {log}"

rule apply_on_data:
    input: str(paths.ss_classifier_model_file), str(paths.B_classifier_model_file)
    output: str(paths.data_testing_B_ProbBs_file)
    log: str(paths.logs_dir / "apply_on_data.log")
    threads: 40
    resources:
        MaxRunHours=8,
        request_memory=150*1024, # in MB
        request_gpus=0 
    shell: "python apply_on_data.py -t {threads} &> {log}"


rule test_on_data:
    input: str(paths.data_testing_B_ProbBs_file), str(paths.bkg_bdt_model_file)
    output: str(paths.data_testing_plots_file)
    log: str(paths.logs_dir / "test_on_data.log")
    threads: 40
    resources:
        MaxRunHours=3,
        request_memory=50*1024, # in MB
        request_gpus=0 
    shell: "python test_on_data.py -t {threads} &> {log}"