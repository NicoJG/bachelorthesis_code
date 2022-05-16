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
from utils.merge_pdfs import merge_pdfs

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

mc_files = paths.B2JpsiKS_mc_files
data_files = paths.B2JpsiKS_data_files

output_dir = paths.data_testing_plots_dir

if paths.data_testing_plots_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir()

data_tree_key = "Bd2JpsiKSDetached/DecayTree"
data_tree_keys = [data_tree_key]*len(data_files)

mc_tree_key = "Bd2JpsiKSDetached/DecayTree"
mc_tree_keys = [mc_tree_key]*len(mc_files)

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
# Load the mc event data for the tail fits
print("Load the event mc data for the tail fits...")
df_mc = load_and_merge_from_root(mc_files, mc_tree_keys, 
                                features=event_features_to_load,
                                cut="(N!=0)&(B_BKGCAT==0)",
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
M_Bd = 5279.65 # MeV
M_Bs = 5366.88 # 

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
plt.axvline(M_Bd, linestyle="dashed", color="grey", label="Bd mass")
plt.axvline(M_Bs, linestyle="dotted", color="grey", label="Bs mass")
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
from utils.histograms import calc_pull
from scipy.stats import norm, expon, crystalball
from iminuit.cost import LeastSquares
from iminuit import Minuit

print("Fit and Plot...")

# %%
# Fit PDF Functions
M_Bd = 5279.65 # MeV
M_Bs = 5366.88 # MeV

def pdf_bkg(x, lambda_bkg, lambda_bkg2, f_bkg):
    pdf_bkg1 = np.exp(-lambda_bkg * x)
    pdf_bkg2 = np.exp(-lambda_bkg2 * x)
    return f_bkg*pdf_bkg1 + (1-f_bkg)*pdf_bkg2

def pdf_B(x, mu_B, sigma_B, sigma_B2, sigma_B3, beta_B,beta_B2, m_B, m_B2, f_B, f_B2):
    pdf_B1 = crystalball.pdf((x-mu_B)/sigma_B, beta_B, m_B)
    pdf_B2 = crystalball.pdf(-(x-mu_B)/sigma_B2, beta_B2, m_B2)
    pdf_B3 = norm.pdf(x, mu_B, sigma_B3)
    return f_B*f_B2*pdf_B1 + (1-f_B)*f_B2*pdf_B2 + (1-f_B)*(1-f_B2)*pdf_B3

def pdfs(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd, beta_Bd2, m_Bd,m_Bd2, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs):
    mu_Bs = mu_Bd + (M_Bs-M_Bd)
    pdf_bkg_ = pdf_bkg(x, lambda_bkg, lambda_bkg2, f_bkg)
    pdf_Bd_ = pdf_B(x, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd,beta_Bd2, m_Bd, m_Bd2, f_Bd, f_Bd2)
    pdf_Bs_ = pdf_B(x, mu_Bs, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd,beta_Bd2, m_Bd, m_Bd2, f_Bd, f_Bd2)
    
    return [N_bkg * pdf_bkg_ , N_Bd * pdf_Bd_ , N_Bs * pdf_Bs_]

def pdf(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd, beta_Bd2, m_Bd,m_Bd2, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs):
    pdfs_ = pdfs(x, lambda_bkg, lambda_bkg2, f_bkg, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd, beta_Bd2, m_Bd,m_Bd2, f_Bd, f_Bd2, N_bkg, N_Bd, N_Bs)
    return np.sum(pdfs_, axis=0)

# %%    
# set the binning
x_min = 5170. # MeV
x_max = 5450. # MeV
n_bins = 150

bin_edges = np.linspace(x_min, x_max, n_bins+1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
bin_widths = np.diff(bin_edges)

# set the file path
fits_dir = output_dir/"fits"
if fits_dir.is_dir():
    shutil.rmtree(fits_dir)
fits_dir.mkdir(exist_ok=True)

# %%
# Fit the tails on mc data
def pdf_B_mc(x, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd,beta_Bd2, m_Bd, m_Bd2, f_Bd, f_Bd2, N):
    return N*pdf_B(x, mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd,beta_Bd2, m_Bd, m_Bd2, f_Bd, f_Bd2)

start_vals_mc = {
    "mu_Bd" : M_Bd,
    "sigma_Bd" : 20,
    "sigma_Bd2" : 30,
    "sigma_Bd3" : 30,
    "beta_Bd" : 0.1,
    "beta_Bd2" : 0.1,
    "m_Bd" : 2,
    "m_Bd2" : 3,
    "f_Bd" : 0.3,
    "f_Bd2" : 0.8,
    "N" : 10000
}

param_limits_mc = {
    "mu_Bd" : (5275, 5282),
    "sigma_Bd" : (5,50),
    "sigma_Bd2" : (5,50),
    "sigma_Bd3" : (5,50),
    "beta_Bd" : (0,5),
    "beta_Bd2" : (0,5),
    "m_Bd" : (1,4),
    "m_Bd2" : (1,4),
    "f_Bd" : (0,1),
    "f_Bd2" : (0,1),
    "N" : (1,np.Infinity)
}

x = bin_centers
bin_counts, _ = np.histogram(df_mc["B_M"], bins=bin_edges)
least_squares = LeastSquares(bin_centers, bin_counts, bin_counts**0.5, pdf_B_mc)
minuit_mc = Minuit(least_squares, **start_vals_mc)

for param, limits in param_limits_mc.items():
    minuit_mc.limits[param] = limits
    
minuit_mc.migrad(ncall=10000)
# %%
# Plot the MC Fit
minuit_mc.hesse()

fit = pdf_B_mc(bin_centers, *minuit_mc.values)
mu_Bd, sigma_Bd, sigma_Bd2, sigma_Bd3, beta_Bd,beta_Bd2, m_Bd, m_Bd2, f_Bd, f_Bd2, N = minuit_mc.values

fit_Bd1 = N*f_Bd*f_Bd2*crystalball.pdf((x-mu_Bd)/sigma_Bd, beta_Bd, m_Bd)
fit_Bd2 = N*(1-f_Bd)*f_Bd2*crystalball.pdf(-(x-mu_Bd)/sigma_Bd2, beta_Bd2, m_Bd2)
fit_Bd3 = N*(1-f_Bd)*(1-f_Bd2)*norm.pdf(x, mu_Bd, sigma_Bd3)

fig = plt.figure(figsize=(5, 7))
fig.suptitle("Fit of the MC B mass")

ax = plt.subplot2grid(shape=(4,1), loc=(0,0), rowspan=3)
ax_pull = plt.subplot2grid(shape=(4,1), loc=(3,0), rowspan=1)

ax.errorbar(bin_centers, bin_counts, yerr=bin_counts**0.5, xerr=bin_widths/2, fmt="none", color="black", label="Data", elinewidth=1.0)

ax.plot(x, fit, label="Fit")
ax.plot(x, fit_Bd1, label="Fit Bd1", alpha=0.4)
ax.plot(x, fit_Bd2, label="Fit Bd2", alpha=0.4)
ax.plot(x, fit_Bd3, label="Fit Bd3", alpha=0.4)
    
ax.set_ylim(bottom=np.min(bin_counts[bin_counts>0])-1, top=np.max(bin_counts)+1)
ax.set_yscale("log")

pull = calc_pull(bin_counts, fit, bin_counts**0.5, 0)

ax_pull.axhline(0, color="black", alpha=0.5)
ax_pull.hist(bin_centers, weights=pull, bins=bin_edges, histtype="stepfilled")

ax_pull.set_xlabel("B mass")
ax_pull.set_ylabel("pull")
ax.set_ylabel("counts")

ax.legend()

fig.tight_layout()
fig.savefig(fits_dir/"000_MC_fit.pdf")
plt.show()
plt.close()


# %%
# Fit Functions
def do_fit(x, y, start_vals, param_limits, fixed_params):
    least_squares = LeastSquares(x, y, y**0.5, pdf)
    m = Minuit(least_squares, **start_vals)
    
    for param, limits in param_limits.items():
        m.limits[param] = limits
        
    for param in fixed_params:
        m.fixed[param] = True
        
    m.migrad(ncall=10000)
    m.hesse()
    
    return m

 # Plot the fit
def plot_fit(minuit, bin_centers, bin_edges, bin_counts, plot_title, file_path):
    x_min = bin_edges[0]
    x_max = bin_edges[-1]
    x = bin_centers
    
    fit = pdf(x, *minuit.values)
    fits = pdfs(x, *minuit.values)
    fit_labels = ["Fit BKG", "Fit Bd", "Fit Bs"]
    
    fig = plt.figure(figsize=(5, 7))
    fig.suptitle(plot_title)
    
    ax = plt.subplot2grid(shape=(4,1), loc=(0,0), rowspan=3)
    ax_pull = plt.subplot2grid(shape=(4,1), loc=(3,0), rowspan=1)
    
    ax.errorbar(bin_centers, bin_counts, yerr=bin_counts**0.5, xerr=bin_widths/2, fmt="none", color="black", label="Data", elinewidth=1.0)
    ax.plot(x, fit, label="Fit")
    for fit_, fit_label in zip(fits, fit_labels):
        ax.plot(x, fit_, label=fit_label)
        
    ax.set_ylim(bottom=np.min(bin_counts[bin_counts>0])-1, top=np.max(bin_counts)+1)
    ax.set_yscale("log")

    pull = calc_pull(bin_counts, fit, bin_counts**0.5, 0)

    ax_pull.axhline(0, color="black", alpha=0.5)
    ax_pull.hist(bin_centers, weights=pull, bins=bin_edges, histtype="stepfilled")
    
    ax_pull.set_xlabel("B mass")
    ax_pull.set_ylabel("pull")
    ax.set_ylabel("counts")

    ax.legend()
    
    fig.tight_layout()
    fig.savefig(file_path)
    #plt.show()
    plt.close()
    
def trapezoidal_rule(x, f):
    a = x[:-1]
    f_a = f[:-1]    
    b = x[1:]
    f_b = f[1:]
    
    return np.sum((b - a)*(f_a + f_b)/2)

def calc_yields(minuit, x):
    fits = pdfs(x, *minuit.values)
    
    yields = []
    
    for fit in fits:
        yields.append(trapezoidal_rule(x, fit))
        
    return yields

# %%
# Fit Params
mc_params = minuit_mc.values

start_vals = {
    "lambda_bkg" : 10**-5,
    "lambda_bkg2" : 10**-3,
    "f_bkg" : 0.5,
    "mu_Bd" : M_Bd,
    "sigma_Bd" : 20,
    "sigma_Bd2" : 20,
    "sigma_Bd3" : 20,
    "beta_Bd" : mc_params["beta_Bd"],
    "beta_Bd2" : mc_params["beta_Bd2"],
    "m_Bd" : mc_params["m_Bd"],
    "m_Bd2" : mc_params["m_Bd2"],
    "f_Bd" : mc_params["f_Bd"],
    "f_Bd2" : mc_params["f_Bd2"],
    "N_bkg" : 10000,
    "N_Bd" : 1000,
    "N_Bs" : 100
}

param_limits = {
    "lambda_bkg" : (10**-7, 0.01),
    "lambda_bkg2" : (10**-7, 0.01),
    "mu_Bd" : (5275, 5282),
    "sigma_Bd" : (5,50),
    "sigma_Bd2" : (5,50),
    "sigma_Bd3" : (5,50),
    "f_bkg" : (0,1),
    "N_bkg" : (1,np.Infinity),
    "N_Bd" : (1,np.Infinity),
    "N_Bs" : (1,np.Infinity)
}

fixed_params = ["beta_Bd", "beta_Bd2", "m_Bd", "m_Bd2", "f_Bd", "f_Bd2"]

# %%
# Do multiple fits
quantiles = np.linspace(0,1,51)
cuts = np.quantile(df_cut.query(f"(B_M>={x_min})&(B_M<={x_max})")["B_ProbBs"], quantiles)
cut_queries = [f"B_ProbBs>={cut:.5f}" for cut in cuts[:-1]]
cut_queries += [f"B_ProbBs<={cut:.5f}" for cut in cuts[1:]]

results = []


for i,cut_query in enumerate(tqdm(cut_queries)):
    bin_counts, _ = np.histogram(df_cut.query(cut_query)["B_M"], bins=bin_edges)
    
    if all(bin_counts<=1):
        print(f"No bin_counts for '{cut_query}'")
        continue
    
    if i>0:
        for key in start_vals.keys():
            start_vals[key] = minuit.values[key]
    
    minuit = do_fit(bin_centers, bin_counts, start_vals, param_limits, fixed_params)
    
    yields = calc_yields(minuit, bin_centers)

    plot_fit(minuit, bin_centers, bin_edges, bin_counts, 
             plot_title=f"Fit of the B mass with '{cut_query}' (valid: {minuit.valid})",
             file_path=fits_dir/f"{i:03d}.pdf")
    
    results.append({
        "n_bkg" : yields[0],
        "n_Bd" : yields[1],
        "n_Bs" : yields[2],
        "cut_query" : cut_query,
        "is_cut_greater" : ">" in cut_query,
        "cut" : float(cut_query[10:]),
        "valid_fit" : minuit.valid
    })
    
# %%
# Merge the fit plots
merge_pdfs(fits_dir, paths.plots_dir/"fits_test_on_data.pdf")
    
# %%
#
df_yields = pd.DataFrame(results)
print(df_yields)

df_yields["n_Bs/n_Bd"] = df_yields["n_Bs"] / df_yields["n_Bd"]
df_yields["n_Bs/all_Bs"] = df_yields["n_Bs"] / df_yields.loc[0,"n_Bs"]
df_yields["n_Bd/all_Bd"] = df_yields["n_Bd"] / df_yields.loc[0,"n_Bd"]

# %%
df_yields_less = df_yields.query("is_cut_greater==False")
df_yields_greater = df_yields.query("is_cut_greater==True")

# %%
# plot the cuts for Bs
fits_res_dir = output_dir/"fit_results"
if fits_res_dir.is_dir():
    shutil.rmtree(fits_res_dir)
fits_res_dir.mkdir(exist_ok=True)

for is_cut_greater in [True, False]:
    for i, vars in enumerate([["n_bkg", "n_Bd"],["n_Bs"],
                 ["n_Bs/n_Bd"],
                 ["n_Bs/all_Bs", "n_Bd/all_Bd"]]):
        temp_df = df_yields.query(f"is_cut_greater=={str(is_cut_greater)}")
        temp_df_invalid = temp_df.query("valid_fit==False")
        sign = ">" if is_cut_greater else "<"
        fig = plt.figure(figsize=(8,6))
        
        for var in vars:
            plt.plot(temp_df["cut"], temp_df[var], ".", label=f"{var} (ProbBs{sign}=cut)")
            #plt.plot(temp_df_invalid["cut"], temp_df_invalid[var], ".", label=f"{var} (ProbBs{sign}=cut) invalid", color="red")
            
        plt.xlabel("cut")
        plt.ylabel("yield")
        #plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.show()
        if is_cut_greater:
            fig.savefig(fits_res_dir/f"01_greater_{i:02d}.pdf")
        else:
            fig.savefig(fits_res_dir/f"01_less_{i:02d}.pdf")
        plt.close()

# %%
# Plot a "ROC-Curve"
fig = plt.figure(figsize=(8,6))
plt.title("similar to a ROC Curve")
for is_cut_greater in [True,False]:
    temp_df = df_yields.query(f"is_cut_greater=={str(is_cut_greater)}")
    sign = ">" if is_cut_greater else "<"
    plt.plot(temp_df["n_Bs/all_Bs"], temp_df["n_Bd/all_Bd"], ".", label=f"ProbBs{sign}=cut")
    
plt.xlabel("n_Bs/all_Bs")
plt.ylabel("n_Bd/all_Bd")
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(fits_res_dir/"02_roc.pdf")
plt.close()
    

# %%
# Merge the plot pdfs
merge_pdfs(fits_res_dir, paths.plots_dir/"fit_res_test_on_data.pdf")
merge_pdfs(output_dir, paths.data_testing_plots_file)

# %%
