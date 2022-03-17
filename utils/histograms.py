from sympy import isprime
import numpy as np

def find_good_binning(x_raw, n_bins_max=50, lower_quantil=0.01, higher_quantil=0.99):
    n_bins = n_bins_max

    x_min = np.quantile(x_raw, lower_quantil)
    x_max = np.quantile(x_raw, higher_quantil)

    bin_edges = np.linspace(x_min, x_max, n_bins_max+1)

    # better binning for integer values (workaround)
    # it works but it's not good 
    # test the data only contains integers
    if np.all(x_raw % 1 == 0):
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

    return bin_edges, bin_centers