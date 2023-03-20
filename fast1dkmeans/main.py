import numpy as np
from numba import njit
from fast1dkmeans.kmeans import cluster_xi, cluster_xi_space
from fast1dkmeans.regularized_kmeans import binary_search


def cluster(x, k, method="binary-search-interpolation", **kwargs):
    """Perform optimal 1d k-means clustering using different methods
    Implemented are
    - dynamic-programming-kn [O(kn), section 2.2, no space saving]
    - dynamic-programming-space [O(kn), section 3, "dp-linear"]
    - binary-search-normal [O(n lg(U) ), section 2.4, "wilber-binary"]
    - binary-search-interpolation [O(n lg(U) ), section 5.2, "wilber-interpolation"]

    The sections all refer to Gronlund et. al (2017)
    "Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D"
    """

    assert method in ("binary-search-interpolation",
                      "binary-search-normal",
                      "dynamic-programming-kn",
                      "dynamic-programming-space",
                      "dynamic-programming"), f"wrong method string provided {method}"

    if method == "dynamic-programming":
        method = "dynamic-programming-space"
    
    x = np.squeeze(np.asarray(x))
    assert len(x.shape)==1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(x), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"

    order = np.argsort(x)
    x = np.array(x, dtype=np.float64)[order]

    if method == "binary-search-interpolation":
        clusters = binary_search(x, k, method=1, **kwargs)
    elif method == "binary-search-normal":
        clusters = binary_search(x, k, method=0, **kwargs)
    elif method == "dynamic-programming-kn":
        clusters = cluster_xi(x, k)
    elif method == "dynamic-programming-space":
        clusters = cluster_xi_space(x, k)
    return undo_argsort(clusters, order)
    

def undo_argsort(sorted_arr, order):
    revert = np.empty_like(order)
    revert[order]=np.arange(len(sorted_arr))
    return sorted_arr[revert]

@njit(cache=True)
def undo_argsort_numba(sorted_arr, order):
    out = np.empty_like(sorted_arr)
    for i, val in enumerate(order):
        out[val] = sorted_arr[i]
    return out

