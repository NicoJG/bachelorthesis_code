from sympy import isprime
import numpy as np

def find_good_binning(fprops, n_bins_max=50, lower_quantile=0.01, higher_quantile=0.99, allow_logx=True, force_logx=False, is_inv_logx=False):
    """Find bin edges for a feature based on the feature properties dictionary
    Only works for numerical features

    Args:
        fdata (np.ndarray or pd.Series): feature data
        fprops (dict): feature properties
        n_bins_max (int, optional): How many bins there should be maximally, because of int binning. Defaults to 50.
        lower_quantile (float, optional): which quantile the lower end of the binning is. Defaults to 0.01.
        higher_quantile (float, optional): which quantile the higher end of the binning is. Defaults to 0.01.
        allow_logx (bool, optional): If a logarithmic x axis is allowed. Defaults to True.
        force_logx (bool, optional): If every numerical feature should be plotted with log x-axis

    Returns:
        bin_edges (np.ndarray), bin_centers(np.ndarray), is_logx (boot)
    """
    assert fprops["feature_type"] == "numerical", "The feature must be a numerical feature for find_good_binning"

    int_only = fprops["int_only"]
    
    n_bins = n_bins_max

    x_min = fprops[f"quantile_{lower_quantile}"]
    x_max = fprops[f"quantile_{higher_quantile}"]
    
    is_logx = (fprops["logx"] and allow_logx) or (force_logx and x_min > 0.0)
    is_logx = is_logx and not int_only

    # better binning for integer values (workaround)
    # it works but it's not good 
    if int_only:
        n_bins = np.abs(x_max-x_min).astype(int) + 1
        # primes are not divisible...
        while isprime(n_bins) and n_bins > n_bins_max:
            n_bins = n_bins + 1
            x_max = x_max + 1
        # divide n_bins until n_bins <= n_bins_max, but only when it is a true divisor (without rest)
        if n_bins > n_bins_max:
            for d in range(1, n_bins+1):
                if n_bins % d == 0 and n_bins / d <= n_bins_max:
                    n_bins = n_bins // d
                    break
            else:
                raise ValueError(f"Could not find a suitable N_bins")
        
        # center bins around the integers
        x_min -= 0.5
        x_max += 0.5
        
    if is_logx and not is_inv_logx:
        x_min = np.log10(x_min)
        x_max = np.log10(x_max)
    if is_logx and is_inv_logx:
        temp_x_min = x_min
        temp_x_max = x_max
        x_max = np.log10(np.ceil(temp_x_max)-temp_x_min)
        x_min = np.log10(np.ceil(temp_x_max)-temp_x_max)
    
    bin_edges = np.linspace(x_min, x_max, n_bins+1)

    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

    if int_only:
        bin_centers = bin_centers.astype(int)

    return bin_edges, bin_centers, is_logx
    
    
def get_hist(x, bin_edges=None, normed=False, is_categorical=False, categorical_values=None):
    if is_categorical:
        assert categorical_values is not None, "Please provide categorical_values for a categorical feature"
        
        x_counts = np.array([(x==cval).sum() for cval in categorical_values])
    else:
        assert bin_edges is not None, "Please provide bin_edges for a numerical feature"
        
        x_counts, _ = np.histogram(x, bins=bin_edges)
    
    sigma_counts = np.sqrt(x_counts)

    if not normed:
        return x_counts, sigma_counts

    if is_categorical:
        bin_widths = np.ones_like(x_counts)
    else:
        bin_widths = np.diff(bin_edges)
    
    x_normed = x_counts / np.dot(x_counts, bin_widths)
    mask = x_counts > 0
    sigma_normed = np.zeros_like(sigma_counts)
    sigma_normed[mask] = (sigma_counts[mask]/x_counts[mask]) * x_normed[mask]
    
    return x_normed, sigma_normed

def calc_pull(x0, x1, sigma0, sigma1):
    x_delta = x0 - x1
    sigma_delta = np.sqrt(sigma0**2 + sigma1**2)

    pull = np.zeros_like(x_delta)
    mask = sigma_delta != 0
    pull[mask] = x_delta[mask] / sigma_delta[mask]

    return pull