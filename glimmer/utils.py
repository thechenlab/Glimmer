import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from typing import List, Tuple
from .model import train_neighbor_weights


### Smooth cluster labels based on spatial k-nearest neighbors (regions)
def spatial_smooth(
    adata: AnnData,
    k: int = 10, 
    spatial_key: str = 'spatial', 
    label_key: str = 'leiden', 
    emb_key: str = 'X_emb_smooth', 
    region_key: str = 'region'
) -> AnnData:
    """Perform spatial label smoothing using k-nearest neighbors.

    This function computes the k-nearest neighbors for each cell based on spatial coordinates
    and assigns the most frequent label among the neighbors as the new label. The smoothed 
    labels are stored in `adata.obs[region_key]`, replacing or creating that column.

    Embeddings specified by `emb_key` are added to the internal DataFrame for reference 
    or potential future use, but are not used in the smoothing calculation.

    Args:
        adata (AnnData): 
            Annotated data object containing spatial coordinates, embeddings, and cell labels.
        k (int, optional): 
            Number of nearest neighbors to use for smoothing. Defaults to 10.
        spatial_key (str, optional): 
            Key in `adata.obsm` where spatial coordinates are stored. Defaults to 'spatial'.
        label_key (str, optional): 
            Key in `adata.obs` containing the original cell labels. Defaults to 'leiden'.
        emb_key (str, optional): 
            Key in `adata.obsm` for embedding vectors to include in the internal DataFrame 
            (not used in smoothing). Defaults to 'X_emb_smooth'.
        region_key (str, optional): 
            Key under which the smoothed labels will be stored in `adata.obs`. 
            The column `adata.obs[region_key]` will be created or overwritten. Defaults to 'region'.

    Returns:
        AnnData: 
            The input AnnData object with a new or updated column `adata.obs[region_key]` 
            containing the smoothed labels.
    """
    # Create base DataFrame with spatial coordinates and cell labels
    df = pd.DataFrame({
        'x': adata.obsm[spatial_key][:, 0], 
        'y': adata.obsm[spatial_key][:, 1],  
        'label': adata.obs[label_key].values 
    })

    # # Add embedding dimensions to the DataFrame
    # for i in range(adata.obsm[emb_key].shape[1]):
    #     df[f'PCA_{i}'] = adata.obsm[emb_key][:, i] 

    # Create DataFrame for embeddings and concatenate all at once
    pca_cols = {f'PCA_{i}': adata.obsm[emb_key][:, i] for i in range(adata.obsm[emb_key].shape[1])}
    df = pd.concat([df, pd.DataFrame(pca_cols)], axis=1)

    # Fit a k-nearest neighbors model on spatial coordinates
    nbrs = NearestNeighbors(n_neighbors=k).fit(df[['x', 'y']])
    _, indices = nbrs.kneighbors(df[['x', 'y']])  

    # Smooth labels by taking the mode of the labels of the k-nearest neighbors
    smoothed_labels = []
    for i in range(df.shape[0]):
        neighbor_labels = df.iloc[indices[i]]['label'].values  
        smoothed_label = pd.Series(neighbor_labels).mode()[0]  
        smoothed_labels.append(smoothed_label) 
    adata.obs[region_key] = smoothed_labels

    return adata


### Generate edge weight distribution from list of spatial weight
def run_logbarrier_weight(
    adata,
    log_barrier_list: List[float] = None,
    k: int = 50,
    feature_emb: str = "X_pca",
    spatial_emb: str = 'spatial', 
    spatial_w: float = 1.0,
    sparsity_w: float = 0.01,
    neighbor_weight: float = 0.1,
    seed: int = 42,
    num_epochs: int = 10000,
    cuda: str = "cuda:0",
    batch: bool = False,
    batch_size: int = 2048
):
    """
    Run a sweep over log_barrier values, train edge weights, and store embeddings.

    Args:
        adata: AnnData object containing PCA and spatial embeddings.
        log_barrier_list: List of log barrier weights to test.
        k: Number of neighbors.
        feature_emb: Key in `obsm` for feature embeddings.
        spatial_emb: Key in `obsm` for spatial coordinates.
        spatial_w: Weight for spatial regularization term.
        sparsity_w: Weight for sparsity regularization.
        neighbor_weight: Initial neighbor weight.
        seed: Random seed.
        num_epochs: Number of training epochs.
        cuda: Device to use for training, e.g., "cuda:0".
        batch: Whether to use minibatch training.
        batch_size: Size of minibatches if used.

    Returns:
        AnnData object with updated weights and embeddings for each log_barrier.
    """
    if log_barrier_list is None:
        log_barrier_list = [0.01, 0.1, 1, 10, 100, 1000]
        print(f"[INFO] log_barrier_list not specified. Defaulting to: {log_barrier_list}")

    for log_barrier in tqdm(log_barrier_list, desc="Sweeping log_barrier"):
        output_emb = f"emb_smooth_{log_barrier}"
        umap_emb = f"umap_{log_barrier}"
        weight_key = f"Weight_{log_barrier}"

        adata = train_neighbor_weights(
            adata, 
            feature_emb=feature_emb, 
            spatial_emb=spatial_emb, 
            output_emb=output_emb,
            k=k, 
            spatial_w=spatial_w, 
            log_barrier_w=log_barrier, 
            sparsity_w=sparsity_w,
            neighbor_weight=neighbor_weight, 
            num_epochs=num_epochs, 
            cuda=cuda, 
            seed=seed, 
            batch=batch, 
            batch_size=batch_size
        )
        adata.obs[weight_key] = adata.obs['Weight']
        adata.obsm[weight_key] = adata.obsm['Weight']
        sc.pp.neighbors(adata, use_rep=output_emb, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
        adata.obsm[umap_emb] = adata.obsm['X_umap']

        print(f"log_barrier = {log_barrier:<7.2g}  mean = {adata.obs['Weight'].mean():<8.4f}  std = {adata.obs['Weight'].std():<8.4f}")

    return adata


# Visualize the distribution of average cell edge weights over varying log barrier strengths
def plot_logbarrier_curve(
    adata,
    log_barrier_list: List[float],
    mean_threshold_high: float = 0.55,
    mean_threshold_low: float = 0.45,
    title: str = "Edge Weight Distribution across Log Barrier Values",
    x_label: str = "Weight of Spatial Information (Log Barrier)",
    y_label: str = "Edge Weight (Mean ± Std)",
    figsize: Tuple[int, int] = (6, 4.5)
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plot the mean ± std of edge weights for each log_barrier value, 
    classify into types, and visualize.

    Args:
        adata: AnnData object with obs['Weight_{log_barrier}'] computed.
        log_barrier_list: List of log_barrier values to analyze.
        mean_threshold_high: Threshold above which a cluster is considered 'regional'.
        mean_threshold_low: Threshold below which a cluster is considered 'cellular'.
        title: Title of the plot.
        figsize: Figure size.

    Returns:
        df: DataFrame with mean/std/type per log_barrier.
        fig: Matplotlib Figure.
        ax: Matplotlib Axes.
    """
    if log_barrier_list is None:
        log_barrier_list = [0.01, 0.1, 1, 10, 100, 1000]
        print(f"[INFO] log_barrier_list not specified. Defaulting to: {log_barrier_list}")
    
    # Step 1: Collect mean/std
    data_list = []
    for lw in log_barrier_list:
        key = f'Weight_{lw}'
        data_list.append({
            'log_barrier': lw,
            'mean': adata.obs[key].mean(),
            'std': adata.obs[key].std()
        })
    df = pd.DataFrame(data_list)

    # Step 2: Assign cluster types
    conditions = [
        df['mean'] > mean_threshold_high,
        df['mean'] <= mean_threshold_low,
        (df['mean'] > mean_threshold_low) & (df['mean'] <= mean_threshold_high)
    ]
    choices = ['regional', 'cellular', 'transitional']
    df['cluster_type'] = np.select(conditions, choices, default='unknown')

    # Step 3: Plot
    color_map = {
        'regional': '#1f77b4', 
        'cellular': '#9467bd', 
        'transitional': '#aec7e8'
    }

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        x, y, err = row["log_barrier"], row["mean"], row["std"]
        ax.plot([x, x], [y - err, y + err], color='grey', linewidth=2.5, alpha=0.7)

    ax.plot(df["log_barrier"], df["mean"], marker='o', linestyle='-', color='black', label="Mean", ms=12)

    for cluster, group in df.groupby("cluster_type"):
        ax.scatter(group["log_barrier"], group["mean"], s=200, color=color_map[cluster], label=cluster, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(df["log_barrier"])
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='cellular', markerfacecolor=color_map['cellular'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='transitional', markerfacecolor=color_map['transitional'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='regional', markerfacecolor=color_map['regional'], markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    return fig, ax