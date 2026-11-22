import numpy as np

from fast1dkmeans import _core


def undo_argsort(sorted_arr, order):
    revert = np.empty_like(order)
    revert[order] = np.arange(len(sorted_arr))
    return sorted_arr[revert]


def cluster(x, k, method="binary-search-interpolation", **kwargs):
    """Perform optimal 1d k-means clustering using different methods
    Implemented are
    - dynamic-programming-kn [O(kn), section 2.2, no space saving]
    - dynamic-programming-space [O(kn), section 3, "dp-linear"]
    - binary-search-normal [O(n lg(U) ), section 2.4, "wilber-binary"]
    - binary-search-interpolation [O(n lg(U) ), section 5.2, "wilber-interpolation"]

    The sections all refer to Gronlund et. al (2017)
    "Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D"

    The binary search methods accept the keyword arguments `max_iter` and `epsilon`.
    """

    assert method in (
        "binary-search-interpolation",
        "binary-search-normal",
        "dynamic-programming-kn",
        "dynamic-programming-space",
        "dynamic-programming",
    ), f"wrong method string provided {method}"

    if method == "dynamic-programming":
        method = "dynamic-programming-space"

    x = np.squeeze(np.asarray(x))
    assert len(x.shape) == 1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(
        x
    ), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"

    order = np.argsort(x)
    x_sorted = np.ascontiguousarray(np.asarray(x, dtype=np.float64)[order])
    k = int(k)

    if method == "binary-search-interpolation":
        clusters = _core.binary_search(x_sorted, k, 1, **kwargs)
    elif method == "binary-search-normal":
        clusters = _core.binary_search(x_sorted, k, 0, **kwargs)
    elif method == "dynamic-programming-kn":
        clusters = _core.dynamic_programming_kn(x_sorted, k, **kwargs)
    else:
        clusters = _core.dynamic_programming_space(x_sorted, k, **kwargs)
    return undo_argsort(clusters, order)
