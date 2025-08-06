import sys
import os
import gc
import time
import torch
import random
import pandas as pd
import scanpy as sc
import numpy as np
import warnings
import random
from tqdm import tqdm
from torch_geometric import seed_everything
from glimmer.model import train_neighbor_weights, set_seed
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# display helper functions
model_idx = int(sys.argv[1])
data_name = sys.argv[2]
seed = int(sys.argv[3])
data_type = sys.argv[4]
device = sys.argv[5]


# Preprocess the data for the models and return the adata object
def adata_preprocess(adata):
    adata.var_names_make_unique()
    adata.obs['n_counts'] = adata.X.sum(1)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.shape[1] > 3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    return adata


# Load Visium data with ground truth labels
def load_visium_data(data_name, path):
    visium_path = path
    file_fold = os.path.join(visium_path + data_name) 
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['ground_truth'] =pd.read_csv(os.path.join(file_fold, "metadata.tsv"), sep='\t')['layer_guess']
    adata = adata[~adata.obs['ground_truth'].isna()]
    return adata


# Load MERFISH data
def load_merfish_data(num, path):
    data_path = path
    data_ids = ["MERFISH_0.04_20241109014506.h5ad",
                "MERFISH_0.14_20241109015203.h5ad",
                "MERFISH_0.24_20241109015212.h5ad",
                "MERFISH_0.09_20241109014907.h5ad",  
                "MERFISH_0.19_20241109015208.h5ad"]
    adata = sc.read_h5ad(data_path + data_ids[int(num)])
    return adata


# Load Slide-seq data
# Subset the Slide-seq data to a circle with a certain radius
def select_cells_within_radius(adata, initial_radius=800, min_cells=30000, max_cells=40000, step_size=50):
    spatial_coords = adata.obsm["spatial"]
    center = spatial_coords.mean(axis=0)
    distances = np.sqrt(((spatial_coords - center) ** 2).sum(axis=1))
    radius = initial_radius
    while True:
        print(f"Trying radius {radius}.")
        selected_cells = np.where(distances <= radius)[0]
        if len(selected_cells) >= min_cells and len(selected_cells) <= max_cells:
            break
        elif len(selected_cells) < min_cells:
            radius += step_size
        else:
            radius -= step_size
            if radius < 0:
                raise ValueError("Cannot find a suitable radius.")
    print(f"Selected {len(selected_cells)} cells within radius {radius}.")
    return adata[selected_cells].copy()

def load_puck_data(num, path):
    slideseq_path = path
    if os.path.exists(slideseq_path+f"Subset_Puck_Num_{num}.h5ad"):
        adata = sc.read_h5ad(slideseq_path+f"Subset_Puck_Num_{num}.h5ad")
        return adata
    else:
        adata = sc.read_h5ad(slideseq_path+f"Puck_Num_{num}.h5ad")
        adata = adata[adata.obs['DeepCCF'] != 'NA']
        adata.obs["ground_truth"] = adata.obs['DeepCCF'].values
        adata.obsm["spatial"] = adata.obs[['Raw_Slideseq_X', 'Raw_Slideseq_Y']].values
        if '_index' in adata.var.columns:
            adata.var.drop('_index', axis=1, inplace=True)
        if adata.raw is not None and '_index' in adata.raw.var.columns:
            adata.raw.var.drop('_index', axis=1, inplace=True)
        adata = select_cells_within_radius(adata)
        adata.write_h5ad(slideseq_path+f"Subset_Puck_Num_{num}.h5ad")
        return adata


# Run GraphST model on MERFISH, Slide-seq, Visium, Slide-tags datasets
def run_GraphST(adata, seed, device, data_type, save_path):
    from GraphST import GraphST
    set_seed(seed)

    if data_type in ('merfish', 'slide_seq'):
        type = 'Slide'
    else:
        type = '10X'

    start_time = time.time()
    print(f"Starting GraphST run {seed} for {data_type}.")
    model = GraphST.GraphST(adata, datatype=type, device=device, random_seed=seed)
    adata = model.train()
    emb = adata.obsm['emb']
    end_time = time.time()
    np.savetxt(f"{save_path}GraphST_emb_seed{seed}.txt", emb, delimiter=",")
    torch.cuda.empty_cache()
    
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of GraphST in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")

    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"GraphST_{data_type}_seed{seed}: {elapsed_time}\n")


# Run STAGATE model on MERFISH, Visium, Slide-tags datasets
def run_STAGATE(adata, seed, device, data_type, save_path):
    import STAGATE_pyG
    from torch_geometric.loader import DataLoader
    import torch.nn.functional as F
    set_seed(seed)
    
    if data_type in ('visium', 'slidetag', 'merfish'):
        r = 150
        start_time = time.time()
        print(f"Starting STAGATE run {seed} for {data_type}.")
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=r)
        adata = STAGATE_pyG.train_STAGATE(adata, device=device, random_seed=seed)
        emb = adata.obsm['STAGATE']
        end_time = time.time()

    elif data_type == 'slide_seq':
        r = 50
        start_time = time.time()
        print(f"Starting STAGATE run {seed} for {data_type}.")
        adata.obs[['X', 'Y']] = adata.obsm['spatial'].astype(float)
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=r)
        data = STAGATE_pyG.Transfer_pytorch_Data(adata)
        data.to(device)
        loader = DataLoader([data], batch_size=2, shuffle=True)
        num_epoch = 1000
        lr = 0.001
        weight_decay = 1e-4
        hidden_dims = [512, 30]
        model = STAGATE_pyG.STAGATE(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in tqdm(range(1, num_epoch+1)):
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                z, out = model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        data.to(device)
        model.eval()
        z, out = model(data.x, data.edge_index)
        emb = z.to('cpu').detach().numpy()
        end_time = time.time()
    else:
        r = 50
        start_time = time.time()
        print(f"Starting STAGATE run {seed} for {data_type} with batches.")
        adata.obs[['X', 'Y']] = adata.obsm['spatial'].astype(float)
        Batch_list = STAGATE_pyG.Batch_Data(adata, num_batch_x=3, num_batch_y=3, spatial_key=['X', 'Y'], plot_Stats=False)
        for temp_adata in Batch_list:
            STAGATE_pyG.Cal_Spatial_Net(temp_adata, rad_cutoff=r)
        data_list = [STAGATE_pyG.Transfer_pytorch_Data(adata) for adata in Batch_list]
        for temp in data_list:
            temp.to(device)
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=r)
        data = STAGATE_pyG.Transfer_pytorch_Data(adata)
        loader = DataLoader(data_list, batch_size=2, shuffle=True)
        num_epoch = 1000
        lr = 0.001
        weight_decay = 1e-4
        hidden_dims = [512, 30]
        model = STAGATE_pyG.STAGATE(hidden_dims = [data_list[0].x.shape[1]] + hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in tqdm(range(1, num_epoch+1)):
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                z, out = model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        data.to(device)
        model.eval()
        z_list = []
        for batch in data_list:
            z, out = model(batch.x, batch.edge_index)
            z_list.append(z.to('cpu').detach().numpy())
        emb = np.concatenate(z_list, axis=0)
        end_time = time.time()

    np.savetxt(f"{save_path}STAGATE_emb_seed{seed}.txt", emb, delimiter=",")
    torch.cuda.empty_cache()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of STAGATE in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"STAGATE_{data_type}_seed{seed}: {elapsed_time}\n")


# Run SPIN model on Slide-seq, MERFISH, Visium, Slide-tags datasets
def run_spin(adata, seed, device, data_type, save_path):
    from spin import spin
    set_seed(seed)

    start_time = time.time()
    print(f"Starting SPIN run {seed} for {data_type}.")
    adata = spin(adata, resolution=0.1, random_state=seed)
    emb = adata.obsm['X_pca_spin']
    end_time = time.time()
    np.savetxt(f"{save_path}SPIN_emb_seed{seed}.txt", emb, delimiter=",")

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of SPIN in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"SPIN_{data_type}_seed{seed}: {elapsed_time}\n")


# Run SpaceFlow model on Slide-seq, MERFISH, Visium, Slide-tags datasets
def run_SpaceFlow(adata, seed, device, data_type, save_path):
    from SpaceFlow import SpaceFlow 
    device = device.split(":")[-1] if "cuda:" in device else device
    set_seed(seed)

    start_time = time.time()
    print(f"Starting SpaceFlow run {seed} for {data_type}.")
    sf = SpaceFlow.SpaceFlow(adata=adata)
    if adata.shape[1] > 3000:
        sf.preprocessing_data(n_top_genes=3000)
    else:
        sf.preprocessing_data(n_top_genes=adata.shape[1])
    sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-3, epochs=1000, 
            max_patience=50, min_stop=100, random_seed=seed, gpu=device, 
            regularization_acceleration=True, edge_subset_sz=1000000)
    emb = sf.embedding
    end_time = time.time()
    np.savetxt(f"{save_path}SpaceFlow_emb_seed{seed}.txt", emb, delimiter=",")

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of SpaceFlow in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"SpaceFlow_{data_type}_seed{seed}: {elapsed_time}\n")


# Run SCAN-IT model on Slide-seq, MERFISH, Visium, Slide-tags datasets
def run_SCANIT(adata, seed, device, data_type, save_path):
    import scanit 
    set_seed(seed)

    start_time = time.time()
    print(f"Starting SCANIT run {seed} for {data_type}.")
    data_slot = 'X_pca' if data_type == 'slide_seq' else None

    scanit.tl.spatial_graph(adata, method='alpha shape', alpha_n_layer=2, knn_n_neighbors=5)
    scanit.tl.spatial_representation(adata, n_h=30, n_epoch=2000, lr=0.001, data_slot = data_slot,
                                     device=device, n_consensus=1, projection='mds', 
                                     python_seed=seed, torch_seed=seed, numpy_seed=seed)
    emb = adata.obsm['X_scanit']
    end_time = time.time()
    np.savetxt(f"{save_path}SCANIT_emb_seed{seed}.txt", emb, delimiter=",")

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of SCANIT in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"SCANIT_{data_type}_seed{seed}: {elapsed_time}\n")


# # Run Our model on Slide-seq, MERFISH, Visium, Slide-tags datasets
def run_Our(adata, seed, device, data_type, save_path):
    set_seed(seed)

    start_time = time.time()
    print(f"Starting Glimmer model run {seed} for {data_type}.")
    k = 50
    adata = train_neighbor_weights(
        adata, 
        feature_emb='X_pca', 
        spatial_emb='spatial', 
        k=k, 
        spatial_w=1, 
        log_barrier_w=5000, 
        neighbor_weight=0.1, 
        num_epochs=10000, 
        cuda = device, 
        seed=seed, 
        batch=False, 
        batch_size = 2048
    )
    emb = adata.obsm['X_emb_smooth']
    end_time = time.time()
    
    np.savetxt(f"{save_path}Our_emb_seed{seed}.txt", emb, delimiter=",")
    torch.cuda.empty_cache()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Duration of Our model in run {seed}: {int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(f"{save_path}run_times.txt", "a") as file:
        file.write(f"Our_{data_type}_seed{seed}: {elapsed_time}\n")


# Main function to run the models
def main(model_idx, data_name, seed, device, data_type):
    set_seed(seed)

    # Define the path prefix
    path_prefix = '/data/qiyu/spatialRegion/benchmark'
    data_paths = {
        'merfish': os.path.join(path_prefix, "Public_data/MERFISH/"),
        'slide_seq': os.path.join(path_prefix, "Public_data/Brain_Slideseq/")
    }

    # Load data
    if data_type not in data_paths:
        raise ValueError(f"Invalid data type: {data_type}. Choose from 'visium', 'merfish', or 'slide_seq'.")
    adata = {
        'merfish': lambda: load_merfish_data(data_name, data_paths['merfish']),
        'slide_seq': lambda: load_puck_data(data_name, data_paths['slide_seq'])
    }[data_type]()
    adata = adata_preprocess(adata)

    # Define model configurations
    models = {
        0: ("Glimmer", run_Our),
        1: ("GraphST", run_GraphST),
        2: ("SPIN", run_spin),
        3: ("STAGATE", run_STAGATE),
        4: ("SpaceFlow", run_SpaceFlow),
        5: ("SCANIT", run_SCANIT)
    }

    if model_idx not in models:
        raise ValueError(f"Invalid model_idx: {model_idx}. Supported indices: {list(models.keys())}")

    # Get model details
    model_name, model_func = models[model_idx]
    save_path = os.path.join(path_prefix, "results", data_type, data_name, model_name + "/")
    os.makedirs(save_path, exist_ok=True)

    # Run the model
    torch.set_num_threads(4)
    torch.cuda.empty_cache()
    model_func(adata, seed, device, data_type, save_path)
    print(f"Finished {model_name} run {seed} for {data_name}.")

if __name__ == "__main__":
    set_seed(seed)
    main(model_idx, data_name, seed, device, data_type)
    
gc.collect()