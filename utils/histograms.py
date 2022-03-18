from sympy import isprime
import numpy as np

def find_good_binning(x_raw, n_bins_max=50, lower_quantil=0.01, higher_quantil=0.99, n_categories_max=10):
    is_categorical = False
    n_bins = n_bins_max

    x_min = np.quantile(x_raw, lower_quantil)
    x_max = np.quantile(x_raw, higher_quantil)

    bin_edges = np.linspace(x_min, x_max, n_bins_max+1)

    only_integers = np.all(x_raw % 1 == 0)

    # check for categorical data
    if len(np.unique(x_raw)) <= n_categories_max:
        is_categorical = True
        bin_centers = np.sort(np.unique(x_raw))
        bin_edges = None
        n_bins = len(bin_centers)

        if only_integers:
            bin_centers = bin_centers.astype(int)

        return bin_edges, bin_centers, is_categorical

    # better binning for integer values (workaround)
    # it works but it's not good 
    # test the data only contains integers
    if only_integers:
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
        bin_edges = np.linspace(x_min-0.5, x_max+0.5, n_bins+1)

    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

    if only_integers:
        bin_centers = bin_centers.astype(int)

    return bin_edges, bin_centers, is_categorical
    
def get_hist(x, bin_edges, normed=False, is_categorical=False, categorical_values=None):
    if is_categorical:
        if categorical_values is None:
            categorical_values, x_counts = np.unique(x, return_counts=True)
        else:
            x_counts = np.array([(x==cval).sum() for cval in categorical_values])
    else:
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