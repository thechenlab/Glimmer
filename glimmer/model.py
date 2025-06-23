import os
import random
import numpy as np
import torch
import scipy.sparse as sp
import torch.optim as optim
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(W, feature_dist, inv_spatial_dist, spatial_w=1, log_barrier_w=50, sparsity_w=0.01):
    r"""
    Compute the regularized Dirichlet energy loss.

    The loss function is defined as:
        \[
        \underset{W \in \mathcal{W}_m}{\text{minimize}} \quad
        \|W \circ Z\|_{1,1} - \alpha \mathbf{1}^\top \log(W \mathbf{1}) + \beta \|W\|_F^2
        \]
    where:
        - \( W \) is the weight matrix representing neighbor weights.
        - \( Z \) is the feature distance matrix scaled by inverse spatial distance, 
            i.e., \( Z = \text{feature\_dist} \circ \text{inv\_spatial\_dist} \).
        - \( \|W \circ Z\|_{1,1} \) is the element-wise product of \( W \) and \( Z \), followed by the L1 norm.
        - \( \alpha \mathbf{1}^\top \log(W \mathbf{1}) \) is the log barrier term ensuring positivity of weights.
        - \( \beta \|W\|_F^2 \) is the Frobenius norm of \( W \), acting as a sparsity penalty.

    Args:
        W (torch.Tensor): 
            Weight matrix representing neighbor weights.
        feature_dist (torch.Tensor): 
            Feature distance matrix between cells and their neighbors.
        inv_spatial_dist (torch.Tensor): 
            Inverse spatial distance matrix between cells.
        spatial_w (float, optional): 
            Weight for spatial distance decay. Defaults to 1.
        log_barrier_w (float, optional): 
            Weight for the log barrier term (\( \alpha \)). Defaults to 50.
        sparsity_w (float, optional): 
            Weight for the sparsity penalty term (\( \beta \)). Defaults to 0.01.

    Returns:
        torch.Tensor: The computed regularized Dirichlet energy loss.
    """
    # Apply spatial decay to the weight matrix W
    W_sigmoid = W * torch.exp(-inv_spatial_dist**spatial_w)
    W_sigmoid = torch.sigmoid(W_sigmoid)

    # Compute the feature distance term
    feature_dist = torch.sum(W_sigmoid * feature_dist * inv_spatial_dist)

    # Compute the log barrier term
    W_log = W_sigmoid.clone()
    W_log[W_log <= 0] = 1e-8 
    log_barrier = -torch.sum(torch.log(W_log))

    # Compute the sparsity penalty term
    sparsity = torch.sum(W_sigmoid**2)

    # Combine all terms to compute the total loss
    total_loss = feature_dist + log_barrier_w * log_barrier + sparsity_w * sparsity

    return total_loss



def train_neighbor_weights(
    adata,
    feature_emb: str = 'X_pca',
    spatial_emb: str = 'spatial',
    output_emb: str = 'X_emb_smooth',
    weight_key: str = 'Weight',
    k: int = 50,
    neighbor_weight: float = 0.1,
    num_epochs: int = 20000,
    lr: float = 1e-4,
    cuda: str = 'cuda:0',
    seed: int = 42,
    batch: bool = False,
    batch_size: int = 2048,
    spatial_w: float = 1,
    log_barrier_w: float = 100,
    sparsity_w: float = 0.01,
    save_neighbor_indices: bool = False
):
    r"""
    Train the neighbor weights using the regularized Dirichlet energy loss.

    Args:
        adata (AnnData): 
            Annotated data object containing feature and spatial embeddings.
        feature_emb (str, optional): 
            Key for feature embedding in `adata.obsm`. Defaults to 'X_pca'.
        spatial_emb (str, optional): 
            Key for spatial coordinates in `adata.obsm`. Defaults to 'spatial'.
        output_emb (str, optional): 
            Key to store the output embedding in `adata.obsm`. Defaults to 'X_emb_smooth'.
        weight_key (str, optional): 
            Key to store the trained weights in `adata.uns` and `adata.obs`. Defaults to 'Weight'.
        k (int, optional): 
            Number of nearest neighbors to consider. Defaults to 50.
        neighbor_weight (float, optional): Weight for neighbor features in smoothing. Defaults to 0.1.
        num_epochs (int, optional): Number of training epochs. Defaults to 20000.
        lr (float, optional): Learning rate for training. Defaults to 1e-4.
        cuda (str, optional): Device to use for training ('cuda:X' or 'cpu'). Defaults to 'cuda:0'.
        batch (bool, optional): Whether to compute feature distances in batches. Defaults to False.
        batch_size (int, optional): Batch size for computing feature distances. Defaults to 2048.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        spatial_w (float, optional): Weight for spatial distance decay. Defaults to 1.
        log_barrier_w (float, optional): Weight for the log barrier term. Defaults to 100.
        sparisty_w (float, optional): Weight for the sparsity penalty term. Defaults to 0.01.

    Returns:
        adata (AnnData): Updated AnnData object with trained weights and smoothed features.
    """
    set_seed(seed)
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')

    # Load feature embeddings
    if feature_emb == 'X':
        if sp.issparse(adata.X):
            feature_x = adata.X.toarray()  
        else:
            feature_x = adata.X  
    else:
        feature_x = adata.obsm.get(feature_emb, None)

    if feature_x is None:
        raise AssertionError(f'Feature embedding {feature_emb} not found in adata.obsm')

    # Load spatial coordinates
    coord = adata.obsm[spatial_emb]
    feature_x = np.ascontiguousarray(feature_x)
    X = torch.tensor(feature_x, device=device).float()

    # Initialize neighbor weight matrix W (num_cells, k)
    W = torch.randn(X.shape[0], k, device=device, requires_grad=True)

    # Compute inverse spatial distance
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coord)
    distances, neighbors = nbrs.kneighbors(coord)
    neighbors = torch.tensor(neighbors, device=device)
    distances = torch.tensor(distances, device=device)
    min_dist = distances.min(dim=1, keepdim=True)[0]
    max_dist = distances.max(dim=1, keepdim=True)[0]
    inv_dist = (distances - min_dist) / (max_dist - min_dist + 1e-8)

    # Compute feature distances (either in batches or all at once)
    if batch:
        num_cells = X.size(0)
        feature_dist = torch.zeros((num_cells, k), device=device)
        for i in range(0, num_cells, batch_size):
            batch_indices = slice(i, min(i + batch_size, num_cells))
            batch_X = X[batch_indices]
            batch_neighbors = neighbors[batch_indices]
            feature_dist[batch_indices] = torch.norm(batch_X.unsqueeze(1) - X[batch_neighbors], dim=-1) ** 2
    else:
        feature_dist = torch.norm(X.unsqueeze(1) - X[neighbors], dim=-1) ** 2

    # Training loop with progress bar
    optimizer = optim.Adam([W], lr=lr)
    with tqdm(range(num_epochs), desc="Training", unit="epoch") as tq:
        for epoch in tq:
            optimizer.zero_grad()
            loss = compute_loss(
                W, 
                feature_dist, 
                inv_dist, 
                spatial_w=spatial_w, 
                log_barrier_w=log_barrier_w, 
                sparisty_w=sparsity_w
            )
            loss.backward()
            optimizer.step()
            epoch_info = f'Loss: {(loss.item()/num_epochs):.4f}' 
            tq.set_postfix_str(epoch_info)

    # Store weights in adata
    W = torch.sigmoid(W).detach().cpu().numpy()
    neighbors = neighbors.cpu().numpy()
    adata.obsm[weight_key] = W
    adata.obs[weight_key] = W.mean(1)
    if save_neighbor_indices:
        adata.uns["neighbor_indices"] = neighbors

    # Smooth the features using trained weights
    weighted_neighbors_features = np.einsum('ij,ijk->ik', W[:, 1:], feature_x[neighbors[:, 1:]])
    smoothed_features = weighted_neighbors_features * neighbor_weight + feature_x * (1 - neighbor_weight)
    adata.obsm[output_emb] = smoothed_features

    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return adata
