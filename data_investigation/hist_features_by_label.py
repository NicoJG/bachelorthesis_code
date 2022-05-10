# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import shutil

# Imports from this project
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import paths
from utils.input_output import load_feature_keys, load_feature_properties, load_preprocessed_data
from utils.histograms import find_good_binning, get_hist, calc_pull
from utils.merge_pdfs import merge_pdfs

# %%
# Constant variables
output_dir = paths.plots_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Customize this to choose what will be plotted 
# required: "label_key", "full_grid"
# optional: "cut_query", "cut_label"
plots_props = [
    {"label_key" : "Tr_is_SS", "full_grid" : True},
    {"label_key" : "B_is_strange", "full_grid" : True},
    {"label_key" : "B_is_strange", "full_grid" : True, "cut_query" : "Tr_is_SS == 1", "cut_label" : "is SS"}
]

# %%
# Read in the feature keys
feature_keys = load_feature_keys(["extracted", "direct","extracted_mc", "direct_mc"])

# Read in the feature properties
feature_props = load_feature_properties()

# %%
# Read the input data
if __name__ == "__main__":
    print("Read in the data...")
    df = load_preprocessed_data(N_entries_max=100000000)
    print("Done reading input")

# %%
# Histogram functions
def prepare_subplots_grid(axes_grid, fkey, draw_pull=True):
    """Returns the figure and axes of what to plot on in a grid depending on which axes types are requested
    
    The full grid would look like:
    | normal | logx          | inv_logx          (3) 
    ---------------------------------------
    | logy   | logx and logy | inv_logx and logy (3) 
    ---------------------------------------
    | pull   | pull_logx     | pull_inv_logx     (1)
    | (1)    | (1)           | (1)

    Arguments:
        axes_grid , list(list(str)): 2d grid of axes and their positions (without pull axes)
            available axes types: "normal", "logx", "logy", "logx_logy", "inv_logx", "inv_logx_logy"
            Every type is allowed only once!
            Example:
            [["normal", "logx"],
             ["logy",   "logx_logy]]
            2nd Example:
            [["normal", "logy"]]
            3rd Example:
            [["inv_logx_logy"]]

    Returns:
        Figure: figure of the plots
        dict(Axes): all generated axes with the corresponding axes type as key
    """    
    
    assert draw_pull, "Not drawing a pull plot is not implemented"
    assert isinstance(fkey, str), "Please provide a string of the feature key"
    assert isinstance(axes_grid, list) and isinstance(axes_grid[0], list) and isinstance(axes_grid[0][0], str), "Please provide a 2d nested list of strings as axes_grid"
    
    axes_grid = np.array(axes_grid)
    
    _, counts =np.unique(axes_grid, return_counts=True)
    assert np.all(counts==1), "Due to the workings of the return dictionary every axes type is allowed only once."
    
    axs = {}
    
    fig_height = 4 * axes_grid.shape[0] + 2
    fig_width = 5 * axes_grid.shape[1]
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    grid_rows = axes_grid.shape[0] * 3 + 1
    grid_cols = axes_grid.shape[1]
    
    # construct the grid of the main axes
    for row in range(axes_grid.shape[0]):
        for col in range(axes_grid.shape[1]):
            axes_type = axes_grid[row, col]
            axs[axes_type] = plt.subplot2grid(shape=(grid_rows, grid_cols),
                                              loc=(row*3, col),
                                              rowspan=3,
                                              colspan=1,
                                              fig=fig)
        
    # construct the remaining grid of the pull axes
    for i, axes_type in enumerate(axes_grid[-1]):
        axs[f"pull_{axes_type}"] = plt.subplot2grid(shape=(grid_rows, grid_cols),
                                          loc=(grid_rows-1, i),
                                          rowspan=1,
                                          colspan=1,
                                          fig=fig)
        
    # set up the y scales
    for axes_type in axs.keys():
        if "logy" in axes_type and not "pull" in axes_type:
            axs[axes_type].set_yscale("log")
    
    # set up the axis labels
    for axes_type in axs.keys():
        if "pull" in axes_type:
            axs[axes_type].set_ylabel("pull")
        else:
            axs[axes_type].set_ylabel("density")
            
        if "inv_logx" in axes_type:
            axs[axes_type].set_xlabel(f"log10( ceil(x_max) - {fkey} )")
        elif "logx" in axes_type:
            axs[axes_type].set_xlabel(f"log10( {fkey} )")
        else:
            axs[axes_type].set_xlabel(f"{fkey}")
    
    return fig, axs


def hist_categorical_feature_by_label(ax, pull_ax, df, fkey, fprops, lkey, lvalues, lvalue_names, 
                                      line_styles=["solid","solid","dotted","dotted","dashed","dashed"],
                                      colors=[f"C{i}" for i in range(6)],
                                      alpha=0.8,
                                      fill=False):
    
    # save the histogrammed values for the pull plot
    x = []
    sigma = []
    
    # get the binning (category values)
    fvalues = fprops["category_values"]
    fvalue_names = fprops["category_names"]
    
    # iterate through each label value
    for lvalue, lvalue_name, ls, c in zip(lvalues, lvalue_names, line_styles, colors):
        x_normed, sigma_normed = get_hist(x=df.query(f"{lkey}=={lvalue}")[fkey],
                                          is_categorical=True,
                                          categorical_values=fvalues,
                                          normed=True)
        
        x.append(x_normed)
        sigma.append(sigma_normed)
        
        if fvalue_names:
            tick_labels = [f"{fval}\n{fval_name}" for fval, fval_name in zip(fvalues, fvalue_names)]
        else:
            tick_labels = [f"{fval}" for fval in fvalues]
            
        # plot the bar plot
        ax.bar(tick_labels, x_normed, yerr=sigma_normed,
               tick_label=tick_labels,
               fill=fill,
               label=lvalue_name,
               edgecolor=c,
               ecolor=c,
               linestyle=ls,
               alpha=alpha)
    
    # plot the pull plot
    if pull_ax is not None:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])
        pull_ax.bar(tick_labels, pull, tick_label=tick_labels, alpha=0.7)
    
    
def hist_numerical_feature_by_label(ax, pull_ax, df, fkey, fprops, lkey, lvalues, lvalue_names,
                                    line_styles=["solid","solid","dotted","dotted","dashed","dashed"],
                                    colors=[f"C{i}" for i in range(6)],
                                    alpha=0.8,
                                    fill=False,
                                    is_logx=False,
                                    is_inv_logx=False,
                                    n_bins_max=300,
                                    lower_quantile=0.0001,
                                    higher_quantile=0.9999):
    # save the histogrammed values for the pull plot
    x = []
    sigma = []
    
    # calculate the binning
    bin_res = find_good_binning(fprops, 
                                n_bins_max=n_bins_max, 
                                lower_quantile=lower_quantile,
                                higher_quantile=higher_quantile,
                                allow_logx=False,
                                force_logx=is_logx,
                                is_inv_logx=is_inv_logx)
    bin_edges, bin_centers, is_logx = bin_res
    
    # iterate through each label value
    for lvalue, lvalue_name, ls, c in zip(lvalues, lvalue_names, line_styles, colors):
        
        query_str = f"({lkey}=={lvalue})"
        query_str += f"&({fkey}>={fprops[f'quantile_{lower_quantile}']})"
        query_str += f"&({fkey}<={fprops[f'quantile_{higher_quantile}']})"
        
        x_raw = df.query(query_str)[fkey]
        
        if is_logx and not is_inv_logx:
            x_raw = np.log10(x_raw)
        elif is_logx and is_inv_logx:
            x_raw = np.log10(np.ceil(fprops[f'quantile_{higher_quantile}'])+10**(-10)-x_raw)
        
        x_normed, sigma_normed = get_hist(x=x_raw, bin_edges=bin_edges, normed=True)
        
        x.append(x_normed)
        sigma.append(sigma_normed)
        
        histtype = "step" if not fill else "stepfilled"
        
        # plot the hist with errorbars
        ax.hist(bin_centers, 
                weights=x_normed, 
                bins=bin_edges, 
                histtype=histtype, 
                label=lvalue_name, 
                color=c,
                linestyle=ls,
                alpha=alpha)
        ax.errorbar(bin_centers, x_normed, 
                               yerr=sigma_normed, 
                               fmt="none", 
                               color=c,
                               alpha=alpha)
    
    # plot the pull plot
    if pull_ax is not None:
        pull = calc_pull(x[0], x[1], sigma[0], sigma[1])
        
        pull_ax.hist(bin_centers, 
                     weights=pull, 
                     bins=bin_edges, 
                     histtype="stepfilled",
                     alpha=0.7)
    

def hist_feature_by_label(df, fkey, fprops, lkey, lprops, full_grid=True, add_cut=False, cut_query=None, cut_label=None, 
                          n_bins_max=300, 
                          lower_quantile=0.0001, 
                          higher_quantile=0.9999):
    
    assert lprops["feature_type"] == "categorical", f"The label ({lkey}) has to be categorical."
    assert fprops["feature_type"] in ["categorical", "numerical"], f"The feature ({fkey}) is not categorical and not numerical..."
    
    lvalues = lprops["category_values"]
    lvalue_names = lprops["category_names"]
    if not lvalue_names:
        lvalue_names = lvalues
    
    if add_cut:
        assert isinstance(cut_query, str), "Please provide a cut_query for pandas.DataFrame.query."
        if cut_label is None:
            cut_label = cut_query
        lvalue_names_cut = [f"{lval_name} ({cut_label})" for lval_name in lvalue_names]
        
    draw_pull = len(lvalues)==2
    
    if fprops["feature_type"]=="numerical":
        x_min = fprops[f"quantile_{lower_quantile}"]
        x_max = fprops[f"quantile_{higher_quantile}"]
        
        # only add logarithmic x axis if the feature is float numerical and x_min>0
        logx_possible = not fprops["int_only"] and x_min > 0.0
        # only add inverse logarithmic x axis if logx_possible and x_max < ceil(x_max)
        inv_logx_possible = logx_possible and x_max < np.ceil(x_max)
        
        # manually allowed inv_logx features:
        allowed_inv_logx = ["Tr_T_PROBNNe", "Tr_T_PROBNNghost", "Tr_T_PROBNNk", "Tr_T_PROBNNmu", "Tr_T_PROBNNp", "Tr_T_PROBNNpi", "Tr_T_TRPCHI2"]
        inv_logx_possible = logx_possible and fkey in allowed_inv_logx
    else:
        logx_possible = False
        inv_logx_possible = False
    
    if full_grid:
        if logx_possible and inv_logx_possible:
            axes_grid = [["normal", "logx", "inv_logx"],
                         ["logy", "logx_logy", "inv_logx_logy"]]
        elif logx_possible:
            axes_grid = [["normal", "logx"],
                         ["logy", "logx_logy"]]
        else:
            axes_grid = [["normal"],
                         ["logy"]]
    elif not full_grid:
        best_axes = fprops["best_axes"]
        if isinstance(best_axes, str):
            axes_grid = [[best_axes]]
        elif isinstance(best_axes[0], str):
            axes_grid = [best_axes]
        else:
            axes_grid = best_axes
        
    fig, axs = prepare_subplots_grid(axes_grid, fkey, draw_pull)

    # set the figure title
    plot_title = f"{fkey} by {lkey}"
    if add_cut:
        plot_title += f" (with: {cut_query})"
    
    if "error_value" in fprops.keys():
        error_val = fprops['error_value']
        error_val_counts = (df[fkey] == error_val).sum()
        error_val_proportion = error_val_counts / df.shape[0]
        plot_title += f"\nwithout the error value {error_val} (proportion: {error_val_proportion*100:.2f}%)"
    fig.suptitle(plot_title)
    
        
    # fill all the axes with the according histogram
    for ax_key in axs.keys():
        if "pull" in ax_key:
            continue
        
        ax = axs[ax_key]
        
        is_logx = "logx" in ax_key
        is_inv_logx = "inv_logx" in ax_key
        
        if f"pull_{ax_key}" in axs.keys():
            pull_ax = axs[f"pull_{ax_key}"]
        else:
            pull_ax = None
            
        if add_cut:
            fill = True
            alpha=0.5
        else:
            fill = False
            alpha=0.8
            
        if fprops["feature_type"] == "categorical":
            hist_categorical_feature_by_label(ax, pull_ax, df, fkey, fprops, lkey, lvalues, lvalue_names, fill=fill, alpha=alpha)
        elif fprops["feature_type"] == "numerical":
            hist_numerical_feature_by_label(ax, pull_ax, df, fkey, fprops, lkey, lvalues, lvalue_names, fill=fill, alpha=alpha, is_logx=is_logx, is_inv_logx=is_inv_logx, n_bins_max=n_bins_max, lower_quantile=lower_quantile, higher_quantile=higher_quantile)
        
        if add_cut:
            if fprops["feature_type"] == "categorical":
                hist_categorical_feature_by_label(ax, pull_ax, df.query(cut_query), fkey, fprops, lkey, lvalues, lvalue_names_cut)
            elif fprops["feature_type"] == "numerical":
                hist_numerical_feature_by_label(ax, pull_ax, df.query(cut_query), fkey, fprops, lkey, lvalues, lvalue_names_cut, is_logx=is_logx, is_inv_logx=is_inv_logx, n_bins_max=n_bins_max, lower_quantile=lower_quantile, higher_quantile=higher_quantile)
        
        if is_inv_logx:
            ax.set_xlabel(f"log10( {np.ceil(x_max)} - {fkey} )")    
        
        if pull_ax is not None:
            if add_cut:
                pull_ax.legend(["all", "cut"], loc="best", fontsize=5)
            if is_inv_logx:
                pull_ax.set_xlabel(f"log10( {np.ceil(x_max)} - {fkey} )")
        
        ax.legend(loc="best")    
    
    fig.tight_layout()
    return fig

# function for multiprocessing
def plot_feature(label_key, output_dir, enumerated_feature_key, full_grid=True, cut_query=None, cut_label=None):
    global df, feature_props
    plot_idx, feature_key = enumerated_feature_key
    output_file = output_dir/f"{plot_idx:03d}_{feature_key}.pdf"
    
    fig = hist_feature_by_label(df, 
                                feature_key, feature_props[feature_key], 
                                label_key, feature_props[label_key],
                                full_grid=full_grid,
                                add_cut=(cut_query is not None),
                                cut_query=cut_query,
                                cut_label=cut_label)
    
    fig.savefig(output_file)
    plt.close(fig)
    
# %%
# Hist of all features by all labels (potentially with a cut)
if __name__ == "__main__":
    for plot_props in plots_props:
        
        label_key = plot_props["label_key"]
        full_grid = plot_props["full_grid"]
        if "cut_query" in plot_props:
            cut_query = plot_props["cut_query"]
            cut_label = plot_props["cut_label"]
        else:
            cut_query = None
            cut_label = None
        
        name = f"features_by_{label_key}"
        if cut_query is not None:
            name += f"_cut_{cut_label.replace(' ', '_')}"
        
        output_label_dir = output_dir / name
        if output_label_dir.is_dir():
            shutil.rmtree(output_label_dir)
        output_label_dir.mkdir(parents=True)
        
        # use multiprocessing to plot all features
        with Pool(processes=50) as p:
            pfunc = partial(plot_feature, label_key, output_label_dir, full_grid=full_grid, cut_query=cut_query, cut_label=cut_label)
            iter = p.imap(pfunc, enumerate(feature_keys))
            pbar_iter = tqdm(iter, total=len(feature_keys), desc=name)
            # start processing, by evaluating the iterator:
            list(pbar_iter)

        merge_pdfs(output_label_dir,  output_dir/f"{name}.pdf")


# %%
