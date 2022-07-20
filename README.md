# Bachelor_Project_Nico_Guth

All the code for my bachelor project at E5a (TU Dortmund). It's about the classification of B_0 and B_s mesons based on the data of the associated tracks without data of the signal decay.



# For using GPUs on the cluster:

As of 06.05.2022 there is no easy way of using GPUs for PyTorch in the E5 cluster.  
- `snakefile --profile htcondor` does not support the ressource `Requirements` so there is no way to control on which machine the job must run  
- `bam` has a GPU that's too old for the PyTorch/CUDA version in `root_forge`  
- `heemskerck` and `beagle` have too new GPUs for the PyTorc/CUDA version in `root_forge` (see below for the fix with conda) 
- `tarek` is compatible with the `root_forge` environment  

The fix for using Snakemake to submit to specific machines requies you to create a new profile to specify `Requirements` in your rule. Example:   
```
rule train:
    input: "<input_path>"
    output: "<output_path>"
    threads: 1
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
        request_gpus=1,
        Requirements='(machine=="beagle.e5.physik.tu-dortmund.de")||(machine=="heemskerck.e5.physik.tu-dortmund.de")'
    shell: "python train.py"
```

The fix for using the GPUs on `heemskerck` and `beagle` is to make a new conda environment.

*IMPORTANT NOTE:* Don't install the environment in your home folder, because this folder is limited to 5GB. 
I noticed that imports take a few seconds if the environment is on `/ceph/...`, but it's better than all the bugs you get otherwise.
*BEFORE* creating the environment do the following:  
```
conda config --add envs_dirs /ceph/users/<username>/.conda/envs
conda config --add pkgs_dirs /ceph/users/<username>/.conda/pkgs
```  

You could use the `environment.yaml` (you should look into it if there are missing or unnecessary packages) from this repository with:  
```
mamba env create -f ~/.../environment.yaml <env_name>
```  
or to update an existing environment:  
```
mamba env update -n <env_name> -f ~/.../environment.yaml
```  

# Getting output from jobs sent to the cluster  

You need to write the output to a logfile and then you can view the tail of the logfile.
Example Rule:
```
rule train:
    input: "<input_path>"
    output: "<output_path>"
    log: "<logfile_path>"
    threads: 20
    resources:
        MaxRunHours=4,
        request_memory=50*1024, # in MB
    shell: "python train.py &> {log}"
```  
While it is running, you can see the output in a different terminal instance with:  
`tail -f <logfile_path>`  
It's important to note that this saves `stdout` and `stderr`. `stderr` is needed because `tqdm` uses this stream.
To print stuff while `tqdm` is running use `print("...", file=sys.stderr)`, otherwise it might be printed only after the progressbar is closed.
If you don't use `tqdm` or anything else that prints to `stderr` just use `>` instead of `&>`.
