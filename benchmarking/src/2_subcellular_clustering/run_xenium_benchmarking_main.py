import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import pickle
import tifffile
from scipy import ndimage
from glimmer.model import *
from glimmer.utils import *
from glimmer.segment import *


# Load cellular data from Cellpose
main_path = "/data/qiyu/spatialRegion"
dir_path = main_path + "/data/xenium/xenium_human_non_diseased_lymph_node/"
output_path = main_path + "/benchmark/Public_data/Xenium_subsets/"

# method for clustering
method = [
    "Glimmer", 
    "Cellpose", 
    "Baysor_SegFree", 
    "Glimmer_CellSegmentation", 
    "Xenium"
][1]

# FOV list for benchmarking
seed = 42
min_transcripts = 5
fov_ranges = [range(7,9), 
              range(9,11), 
              range(10,12), 
              range(13,15), 
              range(15,17), 
              range(17,19), 
              range(21,23), 
              range(23,25), 
              range(27,29), 
              range(29,31), 
              range(31,33), 
              range(33,35), 
              range(39,41), 
              range(41,43), 
              range(43,45)]
fov_names = [f"{r.start}-{r.stop - 1}" for r in fov_ranges]

# leiden resolutions for clustering
resolutions = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


# ------------------------------------------------------------------------------------------------
# Load cellular data from Xenium default
if method == "Xenium":
    cell_adata_path = dir_path + "cell_level_adata_clustered_res1.5.h5ad"
    trans = pd.read_csv(dir_path + "transcripts.csv")
    cells = pd.read_csv(dir_path + "cells.csv.gz", index_col=0)
    valid_cells = cells.index.intersection(trans['cell_id'])   
    trans = trans[trans['cell_id'].isin(valid_cells)]
    cells = cells.loc[valid_cells] 

    if os.path.exists(cell_adata_path):
        adata_all = sc.read_h5ad(cell_adata_path)
    else:
        adata = sc.read_10x_h5(dir_path + "cell_feature_matrix.h5")
        adata = adata[:, ~adata.var_names.str.startswith('NegControlProbe')].copy()
        adata = adata[:, ~adata.var_names.str.startswith('BLANK_')].copy()

        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

        coordinates = cells.loc[adata.obs_names.intersection(cells.index), 
                                ["x_centroid", "y_centroid"]].values
        adata = adata[adata.obs_names.intersection(cells.index), :].copy()
        y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        coordinates[:, 1] = y_max - (coordinates[:, 1] - y_min)
        adata.obsm['X_spatial'] = coordinates

        sc.pp.neighbors(adata, use_rep='X_pca', random_state=seed)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=1.5, random_state=seed)

        adata.write_h5ad(cell_adata_path)

    # Subset a FOV range from all data for clustering
    for fov_range_idx in range(len(fov_ranges)):
        print(f"FOV range: {fov_names[fov_range_idx]}")
        df_save_name = output_path + f"{fov_names[fov_range_idx]}/" + "cellular_leiden_clusters.csv"
        fov = trans['fov_name'].unique()[fov_ranges[fov_range_idx]]
        df = trans[trans['fov_name'].isin(fov)].copy()
        adata = adata_all[adata_all.obs_names.intersection(set(df['cell_id'])), :].copy()

        sc.pp.neighbors(adata, use_rep='X_pca', random_state=seed)
        sc.tl.umap(adata)

        for res in resolutions:
            cluster_key = f"cluster_{res:.2f}"
            sc.tl.leiden(adata, resolution=res, random_state=seed, key_added=cluster_key)
            print(f"Resolution: {res}")
            print(adata.obs[cluster_key].unique())

        cluster_df = adata.obs[[f"cluster_{res:.2f}" for res in resolutions]]
        cluster_df.to_csv(df_save_name, index=True)



# ------------------------------------------------------------------------------------------------
# Load segmentation results from cellpose
def transcript_to_img_coords(df, resolution_um_per_pixel=0.2125):
    df['x_pixel'] = (df['X'] / resolution_um_per_pixel).astype(int)
    df['y_pixel'] = (df['Y'] / resolution_um_per_pixel).astype(int)
    return df

def extract_fov(img, df):
    x_min = df['x_pixel'].min()
    x_max = df['x_pixel'].max()
    y_min = df['y_pixel'].min()
    y_max = df['y_pixel'].max()
    margin_x = int((x_max - x_min) * 0.1)
    margin_y = int((y_max - y_min) * 0.1)
    x_start = max(0, x_min - margin_x)
    x_end = min(img.shape[1], x_max + margin_x)
    y_start = max(0, y_min - margin_y)
    y_end = min(img.shape[0], y_max + margin_y)
    fov = img[y_start:y_end, x_start:x_end]
    return fov, (x_start, y_start, x_end, y_end)

if method == "Cellpose":
    img_path = f"{dir_path}xenium_outs/morphology_focus.ome.tif"
    img = tifffile.imread(img_path, is_ome=True, level=0)
    filter_min_transcripts = True

    for fov_range_idx in range(len(fov_ranges)):
        fov_name = fov_names[fov_range_idx]
        cellpose_folder = f"{output_path}{fov_name}/Cellpose_cyto3_50"
        print(f"FOV range: {fov_name}")

        trans_path = f"{output_path}{fov_name}/transcripts.tsv.gz"
        df = pd.read_csv(trans_path, sep='\t', compression='gzip')
        df = transcript_to_img_coords(df)
        _, (x1, y1, x2, y2) = extract_fov(img, df)

        mask = tifffile.imread(f"{cellpose_folder}/fov_8bit_cp_masks.tif")
        print('number of unique labels:', len(np.unique(mask)))

        fov_img = tifffile.imread(f"{cellpose_folder}/fov_8bit.tif") 
        seg_data = np.load(f"{cellpose_folder}/fov_8bit_seg.npy", allow_pickle=True).item() 

        resolution_um_per_pixel = 0.2125
        df['x_pixel'] = (df['X'] / resolution_um_per_pixel).astype(int) - x1
        df['y_pixel'] = (df['Y'] / resolution_um_per_pixel).astype(int) - y1

        h, w = mask.shape
        df_valid = df[(df['x_pixel'] >= 0) & (df['x_pixel'] < w) & (df['y_pixel'] >= 0) & (df['y_pixel'] < h)].copy()
        df_valid['cell_id'] = mask[df_valid['y_pixel'], df_valid['x_pixel']]
        df_valid = df_valid[df_valid['cell_id'] > 0]
        df_valid.to_csv(f"{cellpose_folder}/cell_transcripts.csv", index=False)

        count_matrix = (
            df_valid.groupby(['cell_id', 'gene'])
            .size()
            .unstack(fill_value=0)
            .astype(int)
        )
        count_matrix = count_matrix.loc[:, ~count_matrix.columns.str.contains('NegControl|BLANK_')]

        coords = ndimage.center_of_mass(np.ones_like(mask), labels=mask, 
                                        index=count_matrix.index.values)
        coords = np.array(coords)
        coords_um = coords * resolution_um_per_pixel

        adata = ad.AnnData(X=count_matrix.values)
        adata.obs_names = count_matrix.index.astype(str)  
        adata.var_names = count_matrix.columns.astype(str) 
        adata.obs['n_umi'] = adata.X.sum(1)
        if filter_min_transcripts:
            adata = adata[adata.obs['n_umi'] > min_transcripts, :].copy()

        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
        sc.pp.neighbors(adata, random_state=seed)
        sc.tl.umap(adata)

        adata_save_name = f"{cellpose_folder}/adata.h5ad"
        for res in resolutions:
            cluster_key = f"cluster_{res:.2f}"
            if cluster_key in adata.obs.keys():
                continue
            sc.tl.leiden(
                adata, 
                resolution=res, 
                random_state=seed, 
                key_added=cluster_key,
                flavor="igraph",          
                directed=False,           
                n_iterations=2
            )
            print(f"Resolution: {res}")
            print(adata.obs[cluster_key].unique())
        adata.write_h5ad(adata_save_name)


# ------------------------------------------------------------------------------------------------
# Clustering for binned data
if method == "Glimmer":

    # load transcripts with nuclei ids from DAPI
    print("Loading transcripts with nuclei ids from DAPI ...")
    trans = pd.read_csv(dir_path + "transcripts_with_nucleus.csv.gz")
    cells = pd.read_csv(dir_path + "cells.csv.gz", index_col=0)
    valid_cells = cells.index.intersection(trans['cell_id'])   
    trans = trans[trans['cell_id'].isin(valid_cells)]

    for fov_range_idx in range(len(fov_ranges)):
        print(f"FOV range: {fov_names[fov_range_idx]}")

        save_dir = output_path + f"{fov_names[fov_range_idx]}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + "Glimmer/")
        
        # filter transcripts with less than min_transcripts
        filter_min_transcripts = True

        # save adata
        adata_save_name = save_dir + "Glimmer/adata.h5ad"
        save_pixle_df = save_dir + "Glimmer/transcripts_bins_mapping.csv"
        save_trans = save_dir + "transcripts.tsv.gz"
        save_feature = save_dir + "features.tsv.gz"

        # Run Glimmer
        if os.path.exists(adata_save_name) and os.path.exists(save_feature) and os.path.exists(save_trans):
            set_seed(seed)
            print(f"Loading adata from {adata_save_name}")
            subset_adata = sc.read_h5ad(adata_save_name)
        else:
            print("Creating files...")
            fov = trans['fov_name'].unique()[fov_ranges[fov_range_idx]]
            df = trans[trans['fov_name'].isin(fov)].copy()
            binned_gex, binned_coord, pixle_df = bin_data(df, n_points=100, seed=42, 
                                                          min_transcripts=5, bin_size=1, 
                                                          step_size=1, return_pixel=True)
            pixle_df.to_csv(save_pixle_df, index=False)
            
            # adata for subset of FOV 
            subset_adata = sc.AnnData(binned_gex)
            subset_adata = subset_adata[:, ~subset_adata.var_names.str.startswith('NegControl')].copy()
            subset_adata = subset_adata[:, ~subset_adata.var_names.str.startswith('BLANK_')].copy()
            subset_adata.obsm['spatial'] = binned_coord.values
            subset_adata.obs['n_umi'] = subset_adata.X.sum(1)

            # filter transcripts with less than min_transcripts
            if filter_min_transcripts:
                subset_adata = subset_adata[subset_adata.obs['n_umi'] > min_transcripts, :].copy()

            # normalize and PCA
            sc.pp.normalize_total(subset_adata, target_sum=1e6)
            sc.pp.log1p(subset_adata)
            sc.tl.pca(subset_adata, svd_solver='arpack', n_comps=50)
            coord = subset_adata.obsm['spatial']
            coord[:, 1] = coord[:, 1].max() - (coord[:, 1] - coord[:, 1].min())
            subset_adata.obsm['spatial'] = coord

            # save adata for our model
            print(f"Saving adata to {adata_save_name}")
            subset_adata.write_h5ad(adata_save_name)

            # save transcripts for benchmarking
            trans_with_cluster = df[df['cell_id'].isin(set(df['cell_id']) & set(cells.index))].copy()
            input_trans = trans_with_cluster[['x_location', 'y_location', 'feature_name']].copy()
            input_trans.columns = ['X', 'Y', 'gene']
            input_trans.sort_values(by='Y', inplace=True)
            input_trans['Count'] = 1 
            feature = input_trans.groupby('gene', as_index=False)['Count'].sum()
            input_trans.to_csv(save_trans, sep='\t', index=False, compression='gzip')
            feature.to_csv(save_feature, sep='\t', index=False, compression='gzip')
            print(f"Files created for {fov_names[fov_range_idx]}")

        # Run our model
        if 'X_emb_smooth' in subset_adata.obsm.keys():
            print("Embedding already exists.")
        else:
            print("Running our model ...")
            subset_adata = train_neighbor_weights(
                subset_adata, 
                feature_emb='X_pca', 
                spatial_emb='spatial', 
                k=15, 
                spatial_w=1, 
                log_barrier_w=200, 
                sparisty_w=0.01,
                neighbor_weight=0.2, 
                num_epochs=10000, 
                cuda='cuda:1', 
                seed=seed, 
                batch=False, 
                batch_size = 2048
            )
            sc.pp.neighbors(subset_adata, use_rep='X_emb_smooth', random_state=seed)
            sc.tl.umap(subset_adata)
        
        for res in resolutions:
            cluster_key = f"cluster_{res:.2f}"
            if cluster_key in subset_adata.obs.keys():
                continue

            sc.tl.leiden(
                subset_adata, 
                resolution=res, 
                random_state=seed, 
                key_added=cluster_key,
                flavor="igraph",          
                directed=False,           
                n_iterations=2
            )
            print(f"Resolution: {res}")
            print(subset_adata.obs[cluster_key].unique())
        subset_adata.write_h5ad(adata_save_name)



# ------------------------------------------------------------------------------------------------
# Segment cells by Voronoi diagram for Glimmer
if method == "Glimmer_CellSegmentation":

    for fov_range_idx in range(len(fov_ranges)):
        print(f"\nFOV range: {fov_names[fov_range_idx]}")

        save_dir = output_path + f"{fov_names[fov_range_idx]}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + "Glimmer/")

        # load adata and transcripts with nuclei ids
        print(f"Loading adata and transcripts with nuclei ids")
        pixle_df = pd.read_csv(save_dir + "Glimmer/transcripts_bins_mapping.csv", index_col=0)
        subset_adata = sc.read_h5ad(save_dir + "Glimmer/adata.h5ad") 

        # build bin metadata
        bin_meta = pd.DataFrame(
            subset_adata.obsm['spatial'], 
            columns=['bin_x_centroid', 'bin_y_centroid'],
            index=subset_adata.obs_names
        )
        bin_meta = pd.concat([bin_meta, subset_adata.obs], axis=1)
        bin_meta = bin_meta.reset_index().rename(columns={'index': 'bin'})

        # merge transcripts with bin metadata
        merged_df_all = pixle_df.merge(bin_meta, on='bin', how='left')

        cluster_gex_list_1 = []
        cluster_gex_list_2 = []
        for res in resolutions:
            # parameters
            min_UMI = 3
            x = "x_location"
            y = "y_location"
            nucleus_label = "nucleus_id"
            cluster_key = f"cluster_{res:.2f}"
            output_col = f"updated_cell_id_1_{res}"
            merged_col = f"merged_cell_id_2_{res}"
            
            print(f"\nRunning cell segmentation under cluster `{cluster_key}`")

            # merge small cell
            merged_df = merged_df_all[merged_df_all[cluster_key].notna()].copy()
            # merged_df['nucleus_id'] = merged_df['nucleus_id'].astype(int).replace(0, np.nan)
            merged_df['nucleus_id'] = merged_df['nucleus_id'].replace(0, pd.NA).astype('Int64')

            result_df = assign_cell_by_voronoi(
                merged_df, x=x, y=y, nucleus_label=nucleus_label, 
                cluster_label=cluster_key, output_col=output_col, 
                n_workers=None, max_distance=None, 
                batch_size=1000, verbose=True)
            result_df_filtered = merge_small_cells(
                result_df, x=x, y=y, cluster=cluster_key, 
                cell_id=output_col, output=merged_col, 
                min_UMI=min_UMI, filter=True, 
                k_neighbors=1, verbose=True) 

            expression_matrix, metadata = build_cell_matrices(
                result_df, cluster_col=cluster_key, 
                x_col=x, y_col=y, gene_col='feature_name', 
                cell_col=output_col) 
            cluster_gex_list_1.append(
                expression_matrix.groupby(metadata['cluster'], 
                                          observed=True).sum())

            expression_matrix, metadata = build_cell_matrices(
                result_df_filtered, cluster_col=cluster_key, 
                x_col=x, y_col=y, gene_col='feature_name', 
                cell_col=merged_col) 
            cluster_gex_list_2.append(
                expression_matrix.groupby(metadata['cluster'], 
                                          observed=True).sum())

        # save cluster gex
        print(f"Saving cluster gex list")
        with open(save_dir + "Glimmer/cluster_gex_1.pkl", 'wb') as f:
            pickle.dump(cluster_gex_list_1, f)
        with open(save_dir + "Glimmer/cluster_gex_2.pkl", 'wb') as f:
            pickle.dump(cluster_gex_list_2, f)



# ------------------------------------------------------------------------------------------------
# Clustering for Baysor
if method == "Baysor_SegFree":
    for fov_name in fov_names:
        loom_path = output_path + f"{fov_name}/output_baysor/ncv_results.loom"
        trans_pixel_df = pd.read_csv(output_path + f"{fov_name}/transcripts.csv")
        pixel_cell_df = ad.read_loom(loom_path).obs.copy()

        # Set index
        trans_pixel_df.index = trans_pixel_df.index.astype(int) + 1
        pixel_cell_df.index = pixel_cell_df.index.astype(float).astype(int)
        data = trans_pixel_df.merge(pixel_cell_df, left_index=True, right_index=True)

        # # cluster by gene expression
        # # Group by ncv_color and aggregate
        # baysor_cellular_data = data.groupby("ncv_color").agg(
        #     cell_x=("X", "mean"),          
        #     cell_y=("Y", "mean"),         
        #     total_transcripts=("gene", "count"),  
        #     gene_counts=("gene", lambda x: x.value_counts().to_dict()) 
        # ).reset_index()

        # expression_matrix = pd.json_normalize(baysor_cellular_data["gene_counts"])
        # expression_matrix = expression_matrix.fillna(0).astype(int)
        # expression_matrix.index = baysor_cellular_data["ncv_color"]
        # cell_coordinates = baysor_cellular_data[["ncv_color", "cell_x", "cell_y"]].set_index("ncv_color")

        # adata = ad.AnnData(expression_matrix)

        # from scipy.sparse import csr_matrix, issparse
        # if issparse(adata.X):
        #     adata.X = csr_matrix(adata.X)

        # adata.obsm["spatial"] = cell_coordinates.to_numpy()
        # sc.pp.normalize_total(adata, target_sum=1e6)
        # sc.pp.log1p(adata)
        # sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
        # sc.pp.neighbors(adata, use_rep='X_pca', random_state=seed)
        # sc.tl.umap(adata)

        # for res in resolutions:
        #     cluster_key = f"cluster_{res:.2f}"
        #     sc.tl.leiden(adata, resolution=res, random_state=seed, key_added=cluster_key)
        #     print(f"Resolution: {res}")
        #     print(adata.obs[cluster_key].unique())
        # adata.write_h5ad(output_path + f"{fov_name}/baysor_segfree_adata.h5ad")

        # cluster by color
        from skimage.color import rgb2lab
        from sklearn.cluster import KMeans

        rgb_values = np.array([
            [
                int(h.lstrip('#')[0:2], 16), 
                int(h.lstrip('#')[2:4], 16),  
                int(h.lstrip('#')[4:6], 16)  
            ] for h in data['ncv_color']
        ])
        lab_values = rgb2lab(rgb_values.astype(np.uint8)[:, np.newaxis, :]).reshape(-1, 3)

        # KMeans clustering
        k_list = [5, 6, 7]
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=42)
            data["color_cluster"] = kmeans.fit_predict(lab_values)  
            data["color_cluster"] = data["color_cluster"].astype('str')
            cluster_to_idx = {cluster: idx for idx, cluster in enumerate(data["color_cluster"].unique())}
            data[f"cluster_k{k}"] = data["color_cluster"].map(cluster_to_idx)

        # save data
        data.to_csv(output_path + f"{fov_name}/baysor_segfree_adata.csv", index=False)
