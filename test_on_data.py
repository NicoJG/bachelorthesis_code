# %%
# Imports
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
import sklearn.metrics as skmetrics
import shutil

# Local Imports
from utils.input_output import load_and_merge_from_root, load_feature_keys
from utils import paths
from utils.histograms import get_hist
#from utils.merge_pdfs import merge_pdfs

# %%
# Constants
parser = ArgumentParser()
parser.add_argument("-t", "--threads", dest="n_threads", default=50, type=int, help="Number of threads to use.")
parser.add_argument("-f", help="Dummy argument for IPython")
args = parser.parse_args()

n_threads = args.n_threads
assert n_threads > 0

N_events_per_dataset = 1000000000000

batch_size_events = 100000

assert paths.data_testing_B_ProbBs_file.is_file(), f"The B classification was not applied yet."
assert paths.bkg_bdt_model_file.is_file(), f"The model '{paths.bkg_bdt_model_file}' does not exist yet."

data_files = paths.B2JpsiKS_data_files

output_dir = paths.data_testing_plots_dir

if paths.data_testing_plots_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir()

data_tree_key = "Bd2JpsiKSDetached/DecayTree"
data_tree_keys = [data_tree_key]*len(data_files)

# %%
# Load all relevant feature keys
print("Load feature keys...")
# Event features:
bdt_features = load_feature_keys(["features_BKG_BDT"], file_path=paths.features_data_testing_file)

lambda_veto_features = load_feature_keys(["features_Lambda_cut"], file_path=paths.features_data_testing_file)

features_other_cuts = load_feature_keys(["features_other_cuts"], file_path=paths.features_data_testing_file)

main_features = load_feature_keys(["main_features"], file_path=paths.features_data_testing_file)

# Load all a list of all existing keys in the data

# Check if all features are in the dataset
event_features_to_load = []
event_features_to_load += bdt_features
event_features_to_load += lambda_veto_features 
event_features_to_load += features_other_cuts
event_features_to_load += main_features

for data_file in data_files:
    with uproot.open(data_file)[data_tree_key] as tree:
        data_all_feature_keys = tree.keys()

    assert len(set(event_features_to_load) - set(data_all_feature_keys))==0, f"The following features are not found in the data: {set(event_features_to_load) - set(data_all_feature_keys)}"

# %%
# Load B_ProbBs
B_ProbBs_dfs = []
with uproot.open(paths.data_testing_B_ProbBs_file) as file:
    for data_file in data_files:
        temp_df = file[data_file.stem].arrays(library="pd")
        if len(B_ProbBs_dfs)!=0:
            temp_df["event_id"] += B_ProbBs_dfs[-1]["event_id"].iloc[-1] + 1
        B_ProbBs_dfs.append(temp_df)

df_B_ProbBs = pd.concat(B_ProbBs_dfs, ignore_index=True)

# %%
####################################
# Start the the analysis of the data
####################################

# %%
# Load the event data for further analysis of the B classification output
print("Load the event data for further analysis of the B classification output...")
df = load_and_merge_from_root(data_files, data_tree_keys, 
                              features=event_features_to_load,
                              cut="N!=0",
                              n_threads=n_threads,
                              N_entries_max_per_dataset=N_events_per_dataset,
                              batch_size=batch_size_events)

# %%
# Remove all events with 0 Tracks, because these events are not in df_tracks
# df.drop(df.query("N==0").index, inplace=True)

# %%
# Check if the event ids match the event ids used for the B classification
assert np.all(df["event_id"]==df_B_ProbBs["event_id"]), "There is a mismatch in the event ids"

# Add the B classification as a feature
df["B_ProbBs"] = df_B_ProbBs["B_ProbBs"]

# %%
# Load and Apply the background bdt
print("Apply BKG BDT...")
# Load the BKG BDT
with open(paths.bkg_bdt_model_file, "rb") as file:
    model_BKG_BDT = pickle.load(file)
    
# Apply the BKG BDT
df["BKG_BDT"] = model_BKG_BDT.predict_proba(df[bdt_features])[:,1]

# %%
# Prepare the veto of the Lambda Background
print("Make the Lambda veto...")

m_pi = 139.57039 # MeV
m_p = 938.272081 # MeV

def weird_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    return ( (E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2 )**0.5

E_piplus = (df["piplus_P"]**2 + m_pi**2)**0.5
E_pplus = (df["piplus_P"]**2 + m_p**2)**0.5
E_piminus = (df["piminus_P"]**2 + m_pi**2)**0.5
E_pminus = (df["piminus_P"]**2 + m_p**2)**0.5

df["m_pi-p+"] = weird_mass(E_piminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_pplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])
df["m_p-pi+"] = weird_mass(E_pminus, df["piminus_PX"], df["piminus_PY"], df["piminus_PZ"], 
                           E_piplus, df["piplus_PX"], df["piplus_PY"], df["piplus_PZ"])

# %%
# Make a veto feature
veto_m_min, veto_m_max = 1095., 1140. 
veto_probnn = 0.10

df["lambda_veto"] = (veto_m_min < df["m_pi-p+"]) & (df["m_pi-p+"] < veto_m_max) & (df["piplus_ProbNNp"] > veto_probnn)
df["lambda_veto"] |= (veto_m_min < df["m_p-pi+"]) & (df["m_p-pi+"] < veto_m_max) & (df["piminus_ProbNNp"] > veto_probnn)

df["lambda_veto"] = df["lambda_veto"].astype(int)

# %%
# Plot the invariant masses for the cut of the Lambda Background
n_bins = 100
x_min, x_max = 1060.0 , 1180.0
bins = np.linspace(x_min, x_max, n_bins+1)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle(r"invariant mass used for the $\Lambda$ veto")

ax0.hist(df["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="no veto")
ax0.hist(df.query("lambda_veto==0")["m_pi-p+"], histtype="step", bins=bins, alpha=0.5, label="with veto")
ax1.hist(df["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="no veto")
ax1.hist(df.query("lambda_veto==0")["m_p-pi+"], histtype="step", bins=bins, alpha=0.5, label="with veto")

ax0.set_xlabel(r"$m(\pi^- p^+)$")
ax1.set_xlabel(r"$m(\pi^+ p^-)$")

ax0.set_ylabel("counts")
ax1.set_ylabel("counts")

ax0.legend()
ax1.legend()
plt.tight_layout()
#plt.show()
plt.savefig(output_dir/"01_lambda_veto.pdf")
plt.close()

# %%
# apply all cuts
print("Apply all cuts...")
cut_query = "(lambda_veto == 0)"
cut_query += "&(BKG_BDT > 0.5)"
cut_query += "&(KS0_TAU > 0.5*10**-3)"
cut_query += "&(B_LOKI_DTF_CHI2NDOF<5)"

#cut_query += "&(piplus_TRACK_Type==5)"
#cut_query += "&(piminus_TRACK_Type==5)"

df_cut = df.query(cut_query)

# %%
# Plot the B classification output 
print("Plot the B classifier output...")
plt.figure(figsize=(8,6))
plt.title("B classifier output distribution on the data (with all cuts)")
plt.hist(df_cut["B_ProbBs"], histtype="step", bins=200, range=(0. , 1.))
plt.xlabel("B classifier output")
plt.ylabel("counts")
#plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir/"03_B_classifier_output.pdf")
plt.close()

# %%
# Plot the B mass with and without the cuts and the B classification
print("Create all the B mass plots before fitting...")

x_min = 5150. # MeV
x_max = 5550. # MeV

n_bins = 200

bin_edges = np.linspace(x_min, x_max, n_bins+1)

# Plot raw B mass distribution
plt.figure(figsize=(8,6))
plt.title("B mass distribution (before any cuts)")
plt.hist(df["B_M"], histtype="step", bins=bin_edges, label="B_M")
plt.xlabel("B mass")
plt.ylabel("counts")
#plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.savefig(output_dir/"04_B_mass_00_raw.pdf")
plt.close()

# Plot B mass distribution after bdt cut
plt.figure(figsize=(8,6))
plt.title("B mass distribution (with cut only on the bkg bdt)")
plt.hist(df.query("(BKG_BDT>=0.5)")["B_M"], histtype="step", bins=bin_edges, label="B_M")
plt.xlabel("B mass")
plt.ylabel("counts")
#plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir/"04_B_mass_01_with_bdt_cut.pdf")
plt.close()

# Plot B mass distribution after all cuts
plt.figure(figsize=(8,6))
plt.title("B mass distribution with all cuts applied (lambda, bkg_bdt, misid)")
plt.hist(df_cut["B_M"], histtype="step", bins=bin_edges, label="B_M")
plt.xlabel("B mass")
plt.ylabel("counts")
#plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.savefig(output_dir/"04_B_mass_02_with_all_cuts.pdf")
plt.close()

# Plot B mass distribution after B classification
m_Bd = 5279.65 # MeV
m_Bs = 5366.88 # 

ProbBs_Bd_max = 0.1
ProbBd_Bs_min = 0.6

x = [df_cut.query(f"B_ProbBs<={ProbBs_Bd_max}")["B_M"], 
     df_cut.query(f"B_ProbBs>={ProbBd_Bs_min}")["B_M"],
     df_cut["B_M"]]
labels = [f"B_ProbBs<={ProbBs_Bd_max}", 
          f"B_ProbBs>={ProbBd_Bs_min}",
          "baseline with bkg cuts"]

plt.figure(figsize=(8,6))
plt.title(r"""B mass distribution of real data ($B^0_{d/s} \rightarrow J/\Psi + K^0_S$)
with cuts and B classification""")
plt.axvline(m_Bd, linestyle="dashed", color="grey", label="Bd mass")
plt.axvline(m_Bs, linestyle="dotted", color="grey", label="Bs mass")
plt.hist(x, histtype="step", density=False, bins=bin_edges, label=labels)
plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.savefig(output_dir/"04_B_mass_03_B_classification.pdf")
plt.close()

print("Plotting done.")

# %%
###################
# Fitting
###################
print("Fit and Plot...")

# %%
# Get the histogram of B_M with all cuts applied (apart from the Bd/Bs classification)
x_min = 5150. # MeV
x_max = 5450. # MeV
n_bins = 150

bin_edges = np.linspace(x_min, x_max, n_bins+1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

bin_widths = np.diff(bin_edges)

bin_counts, _ = np.histogram(df_cut.query("B_ProbBs>=0.5")["B_M"], bins=bin_edges)
#bin_counts, _ = np.histogram(df_cut["B_M"], bins=bin_edges)


# %%
# Do fits

from scipy.stats import norm, expon 
from iminuit.cost import LeastSquares
from iminuit import Minuit

m_Bd = 5279.65 # MeV
m_Bs = 5366.88 # MeV

def pdf_bkg(x, lambda_bkg, lambda_bkg2, f_bkg):
    pdf_bkg1 = np.exp(-lambda_bkg * x)
    pdf_bkg2 = np.exp(-lambda_bkg2 * x)
    return f_bkg*pdf_bkg1 + (1-f_bkg)*pdf_bkg2

def pdf_part(x, sigma_part):
    pdf_part_ = norm.pdf(x, 5100, sigma_part)
    return pdf_part_

def pdf_B(x, mu_B, sigma_B, sigma_B2, sigma_B3, f_B, f_B2):
    pdf_Bd1 = norm.pdf(x, mu_B, sigma_B)
    pdf_Bd2 = norm.pdf(x, mu_B, sigma_B2)
    pdf_Bd3 = norm.pdf(x, mu_B, sigma_B3)
    return f_B*pdf_Bd1 + (1-f_B)*(f_B2)*pdf_Bd2 + (1-f_B)*(1-f_B2)*pdf_Bd3

def pdfs(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs):
    mu_Bs = mu_Bd + (m_Bs-m_Bd)
    pdf_bkg_ = pdf_bkg(x, lambda_bkg, lambda_bkg2, f_bkg)
    pdf_Bd_ = pdf_B(x, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, f_Bd, f_Bd2)
    pdf_Bs_ = pdf_B(x, mu_Bs, sigma_Bd, sigma_Bd2, sigma_Bd3, f_Bd, f_Bd2)
    
    return [N_bkg * pdf_bkg_ , N_Bd * pdf_Bd_ , N_Bs * pdf_Bs_]

def pdf(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs):
    pdfs_ = pdfs(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs)
    return np.sum(pdfs_, axis=0)

least_squares = LeastSquares(bin_centers, 
                             bin_counts, 
                             bin_counts**0.5, 
                             pdf,
                             verbose=0)

m = Minuit(least_squares,
           lambda_bkg = 10**-5,
           lambda_bkg2 = 10**-3,
           f_bkg=0.5,
           mu_Bd=m_Bd,
           sigma_Bd=20,
           sigma_Bd2=30,
           sigma_Bd3=40,
           f_Bd=0.5,
           f_Bd2=0.5,
           N_bkg=10000,
           N_Bd=1000,
           N_Bs=100)

m.limits["lambda_bkg"] = (10**-7, 0.01)
m.limits["lambda_bkg2"] = (10**-7, 0.01)
m.limits["mu_Bd"] = (5275, 5282)
m.limits["sigma_Bd"] = (5, 30)
m.limits["sigma_Bd2"] = (5, 50)
m.limits["sigma_Bd3"] = (5, 50)
m.limits["f_bkg"] = (0,1)
m.limits["f_Bd"] = (0,1)
m.limits["f_Bd2"] = (0,1)
m.limits["N_bkg"] = (1,np.Infinity)
m.limits["N_Bd"] = (1,np.Infinity)
m.limits["N_Bs"] = (1,np.Infinity)



# %%
m.migrad()

# %%
m.hesse()
print("Fitting Done.")

# %%
# Plot the fit
x_lin = np.linspace(x_min, x_max, 1000)

fit = pdf(x_lin, *m.values)

fits = pdfs(x_lin, *m.values)

fit_labels = ["Fit BKG", "Fit Bd", "Fit Bs"]


plt.figure(figsize=(8,6))
plt.errorbar(bin_centers, bin_counts, yerr=bin_counts**0.5, xerr=bin_widths/2, fmt="none", color="black", label="Data", elinewidth=1.0)
plt.plot(x_lin, fit, label="Fit")
for fit_, fit_label in zip(fits, fit_labels):
    plt.plot(x_lin, fit_, label=fit_label)

plt.xlabel("B mass")
plt.ylabel("counts")
plt.yscale("log")

plt.ylim(bottom=np.min(bin_counts[bin_counts>0]), top=np.max(bin_counts))

plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig(output_dir/"05_fits_B_M.pdf")
plt.close()

# %%
#merge_pdfs(output_dir, paths.data_testing_plots_file)

# %%
