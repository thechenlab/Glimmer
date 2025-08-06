"""
This module contains helper functions for the ´benchmarking´ subpackage.
"""

from typing import Optional
import numpy as np
import scanpy as sc
from anndata import AnnData
from scib_metrics.nearest_neighbors import pynndescent
from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix


def compute_knn_graph_connectivities_and_distances(
        adata: AnnData,
        feature_key: str="nichecompass_latent",
        knng_key: str="nichecompass_latent_15knng",
        n_neighbors: int=15,
        random_state: int=0,
        n_jobs: int=1) -> None:
    """
    Compute approximate k-nearest-neighbors graph.

    Parameters
    ----------
    adata:
        AnnData object with the features for knn graph computation stored in
        ´adata.obsm[feature_key]´.
    feature_key:
        Key in ´adata.obsm´ that will be used to compute the knn graph.
    knng_key:
        Key under which the knn graph connectivities  will be stored
        in ´adata.obsp´ with the suffix '_connectivities', the knn graph
        distances will be stored in ´adata.obsp´ with the suffix '_distances',
        and the number of neighbors will be stored in ´adata.uns with the suffix
        '_n_neighbors' .      
    n_neighbors:
        Number of neighbors of the knn graph.
    random_state:
        Random state for reproducibility.   
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    """
    neigh_output = pynndescent(
        adata.obsm[feature_key],
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs)
    indices, distances = neigh_output.indices, neigh_output.distances
    
    # This is a trick to get lisi metrics to work by adding the tiniest possible value
    # to 0 distance neighbors so that each cell has the same amount of neighbors 
    # (otherwise some cells lose neighbors with distance 0 due to sparse representation)
    row_idx = np.where(distances == 0)[0]
    col_idx = np.where(distances == 0)[1]
    new_row_idx = row_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    new_col_idx = col_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    distances[new_row_idx, new_col_idx] = (distances[new_row_idx, new_col_idx] +
                                           np.nextafter(0, 1, dtype=np.float32))

    # modified from sc.neighbors._compute_connectivities_umap (invalid function from scanpy)
    def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
        rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
        cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
        vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

        for i in range(knn_indices.shape[0]):
            for j in range(n_neighbors):
                if knn_indices[i, j] == -1:
                    continue  # We didn't get the full knn for i
                if knn_indices[i, j] == i:
                    val = 0.0
                else:
                    val = knn_dists[i, j]

                rows[i * n_neighbors + j] = i
                cols[i * n_neighbors + j] = knn_indices[i, j]
                vals[i * n_neighbors + j] = val

        result = coo_matrix((vals, (rows, cols)),
                                        shape=(n_obs, n_obs))
        result.eliminate_zeros()
        return result.tocsr()

    def compute_connectivities_umap(
        knn_indices, knn_dists,
        n_obs, n_neighbors, set_op_mix_ratio=1.0,
        local_connectivity=1.0):
        
        X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
        connectivities = fuzzy_simplicial_set(
            X,
            n_neighbors,
            None,
            None,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
        )

        if isinstance(connectivities, tuple):
            connectivities = connectivities[0]

        distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

        return distances, connectivities.tocsr()

    sp_distances, sp_conns = compute_connectivities_umap(
            indices[:, :n_neighbors],
            distances[:, :n_neighbors],
            adata.n_obs,
            n_neighbors=n_neighbors)
    
    adata.obsp[f"{knng_key}_connectivities"] = sp_conns
    adata.obsp[f"{knng_key}_distances"] = sp_distances
    adata.uns[f"{knng_key}_n_neighbors"] = n_neighbors


def convert_to_one_hot(vector: np.ndarray,
                       n_classes: Optional[int]) -> np.array:
    """
    Converts an input 1-D vector of integer labels into a 2-D array of one-hot
    vectors, where for an i'th input value of j, a '1' will be inserted in the
    i'th row and j'th column of the output one-hot vector.
    
    Implementation is adapted from
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L498
    (05.12.22).

    Parameters
    ----------
    vector:
        Vector to be one-hot-encoded.
    n_classes:
        Number of classes to be considered for one-hot-encoding. If ´None´, the
        number of classes will be inferred from ´vector´.

    Returns
    ----------
    one_hot:
        2-D NumPy array of one-hot-encoded vectors.

    Example:
    ´´´
    vector = np.array((1, 0, 4))
    one_hot = _convert_to_one_hot(vector)
    print(one_hot)
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]
    ´´´
    """
    if n_classes is None:
        n_classes = np.max(vector) + 1

    one_hot = np.zeros(shape=(len(vector), n_classes))
    one_hot[np.arange(len(vector)), vector] = 1
    return one_hot.astype(int)
