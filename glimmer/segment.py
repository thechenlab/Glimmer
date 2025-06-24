import math
import zarr
import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree, ConvexHull, Voronoi, KDTree
from zarr import open as open_zarr
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely import wkb
from alphashape import alphashape
from joblib import Parallel, delayed
from matplotlib.colors import Colormap
from typing import Tuple
from scipy.sparse import coo_matrix


### Bin spatial points and compute gene expression per bin (For image-based data)
def bin_spatial_points(
    data: pd.DataFrame,
    n_points: int = 100,
    min_transcripts: int = 5,
    bin_size: int = 1,
    step_size: int = 1,
    x: str = "x_location",
    y: str = "y_location",
    gene: str = "feature_name",
    transcript_id: str = "transcript_id",
    seed: int = 42,
    return_pixel: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """Spatially bin transcriptomic data and aggregate gene expression per bin.

    This function estimates an appropriate bin size by sampling random points and 
    expanding bins until the minimum required transcript count is achieved. It then 
    bins all transcripts, aggregates gene expression, and optionally returns visuals.

    Args:
        data (pd.DataFrame): 
            Input DataFrame containing spatial coordinates (x, y), `gene`, and `transcript_id`.
        n_points (int, optional): 
            Number of random points sampled to estimate bin size. Defaults to 100.
        min_transcripts (int, optional): 
            Minimum number of transcripts required per bin. Defaults to 5.
        bin_size (int, optional): 
            Starting bin size (square width). Defaults to 1.
        step_size (int, optional): 
            Step size to increment bin if transcript count is insufficient. Defaults to 1.
        x (str, optional): 
            Column name for x-coordinate. Defaults to "x_location".
        y (str, optional): 
            Column name for y-coordinate. Defaults to "y_location".
        gene (str, optional): 
            Column name representing gene names. Defaults to "feature_name".
        transcript_id (str, optional): 
            Column name representing unique transcript identifiers. Defaults to "transcript_id".
        seed (int, optional): 
            Random seed used in sampling for reproducibility. Defaults to 42.
        return_pixel (bool, optional): 
            Whether to return the original data with bin assignments. Defaults to True.
        plot (bool, optional): 
            Whether to display a histogram of transcripts per bin. Defaults to False.

    Returns:
        pd.DataFrame: 
            Gene expression matrix aggregated by bin.
        pd.DataFrame: 
            Bin center coordinates for visualization or downstream analysis.
        pd.DataFrame (optional): 
            Original input DataFrame with additional bin assignment columns 
            (only returned if `return_pixel=True`).

    Raises:
        ValueError: If no valid bin size is found satisfying `min_transcripts`.
    """
    # Sample n_points from the data
    sampled_points = data.sample(n=n_points, random_state=seed)

    # Calculate bin sizes for the sampled points
    bin_sizes = []
    for _, row in sampled_points.iterrows():
        current_bin_size = bin_size  # Initialize bin size
        while True:
            # Define the square boundaries for the current bin
            x_all_max, x_all_min = data[x].max(), data[x].min()
            y_all_max, y_all_min = data[y].max(), data[y].min()
            x_min = np.clip(row[x] - current_bin_size / 2, x_all_min, x_all_max)
            x_max = np.clip(row[x] + current_bin_size / 2, x_all_min, x_all_max)
            y_min = np.clip(row[y] - current_bin_size / 2, y_all_min, y_all_max)
            y_max = np.clip(row[y] + current_bin_size / 2, y_all_min, y_all_max)

            # Count the number of transcripts within the square
            count = data[(data[x] >= x_min) & (data[x] <= x_max) &
                         (data[y] >= y_min) & (data[y] <= y_max)].shape[0]

            # If count is sufficient, save bin size
            if count >= min_transcripts:
                bin_sizes.append(current_bin_size)
                break

            # Otherwise increase bin size
            current_bin_size += step_size

    # Ensure bin_sizes is not empty
    if not bin_sizes:
        raise ValueError("No valid bin sizes found. Adjust the parameters.")

    # Compute the mean bin size
    final_bin_size = np.mean(bin_sizes)
    print(f"Averaged bin size: {final_bin_size:.2f}")

    # Assign bins to the data
    bins_x = np.arange(data[x].min(), data[x].max() + final_bin_size + 1e-6, final_bin_size)
    bins_y = np.arange(data[y].min(), data[y].max() + final_bin_size + 1e-6, final_bin_size)

    data = data.copy()
    data["bin_x"] = np.digitize(data[x], bins_x).astype(int)
    data["bin_y"] = np.digitize(data[y], bins_y).astype(int)
    data["bin"] = data["bin_x"].astype(str) + "-" + data["bin_y"].astype(str)

    # Group by bin and compute binned features
    binned_data = data.groupby(["bin_x", "bin_y"]).agg(
        center_x=(x, "mean"), 
        center_y=(y, "mean"),  
        num_transcripts=(transcript_id, "count"),  
        gene_expression=(gene, lambda x: dict(x.value_counts()))  
    ).reset_index()

    # Create consistent bin ID
    binned_data["bin_x"] = binned_data["bin_x"].astype(int)
    binned_data["bin_y"] = binned_data["bin_y"].astype(int)
    binned_data["bin"] = binned_data["bin_x"].astype(str) + "-" + binned_data["bin_y"].astype(str)

    # Create expression matrix
    binned_data = binned_data.set_index("bin")
    binned_expression = pd.DataFrame(binned_data["gene_expression"].tolist()).fillna(0).astype(int)
    binned_expression.index = binned_data.index
    binned_coordinates = binned_data[["center_x", "center_y"]]

    # Plot histogram if needed
    if plot:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(binned_expression.sum(axis=1), bins=50, color="skyblue", edgecolor="black")
        ax.set_title("Histogram of Binned Expression", fontsize=14)
        ax.set_xlabel("Total Transcripts per Bin", fontsize=12)
        ax.set_ylabel("Number of Bins", fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Return results
    if return_pixel:
        return binned_expression, binned_coordinates, data
    else:
        return binned_expression, binned_coordinates


# ------------------------------------------------------------------------------------------------
### Get pixel size from Xenium 
# When using DAPI image to segment nuclei, we need to get the pixel size from the OME-XML file
def get_pixel_size_from_ome(xml_string):
    """
    Extract pixel size from OME-XML string.
    """
    root = ET.fromstring(xml_string)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    pixels = root.find('.//ome:Pixels', namespaces=ns)
    px_size_x = float(pixels.attrib['PhysicalSizeX'])
    px_size_y = float(pixels.attrib['PhysicalSizeY'])
    return px_size_x, px_size_y

def open_zarr(path):
    """
    Open a zarr store from a path.
    """
    store = zarr.ZipStore(path, mode='r') if path.endswith(".zip") else zarr.DirectoryStore(path)
    return zarr.group(store=store)


### assign nucleus ids from DAPI image to transcripts
def assign_nucleus_ids_to_transcripts(
        transcripts_path: str, 
        cells_path: str, 
        trans_df_save_file: str, 
        pixel_size_x_um: float = 0.2125, 
        pixel_size_y_um: float = 0.2125
) -> pd.DataFrame:
    """Assigns nucleus IDs to transcripts using a segmentation mask and spatial coordinates.

    This function reads transcript positions and a cell segmentation mask, maps each 
    transcript to its corresponding nucleus (if any), and saves the updated DataFrame.

    Args:
        transcripts_path (str): 
            Path to the `transcripts.csv.gz` file containing transcript spatial coordinates.
        cells_path (str): 
            Path to the `cells.zarr.zip` file containing labeled nucleus segmentation mask.
        trans_df_save_file (str): 
            File path to save the updated transcript DataFrame with assigned nucleus IDs.
        pixel_size_x_um (float, optional): 
            Microns per pixel in the x-direction. Defaults to 0.2125.
        pixel_size_y_um (float, optional): 
            Microns per pixel in the y-direction. Defaults to 0.2125.

    Returns:
        pd.DataFrame: 
            A DataFrame of transcripts including assigned nucleus IDs.
    """
    # Load transcript data and nucleus segmentation mask
    trans_df = pd.read_csv(transcripts_path)
    nuc_root = open_zarr(cells_path)
    nucseg_mask = np.array(nuc_root["masks"][0])
    H, W = nucseg_mask.shape

    # Convert spatial coordinates from microns to pixels
    x_coords = np.array(trans_df["x_location"])
    y_coords = np.array(trans_df["y_location"]) 
    x_pix = np.round(x_coords / pixel_size_x_um).astype(int)
    y_pix = np.round(y_coords / pixel_size_y_um).astype(int)

    # Filter valid pixel coordinates within image bounds
    valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
    df = pd.DataFrame({
        "x_location": x_coords[valid],
        "y_location": y_coords[valid],
        "nucleus_id": nucseg_mask[y_pix[valid], x_pix[valid]].astype(int)
    })
    print(f"Number of transcripts with nucleus IDs: {df['nucleus_id'].notna().sum()}")
    
    # Merge nucleus ID into transcript dataframe and save
    trans_df = trans_df.merge(df, on=["x_location", "y_location"], how="left")
    trans_df.to_csv(trans_df_save_file, index=False, compression="gzip")
    print(f"Saved transcript dataframe with nucleus IDs to {trans_df_save_file}")


# ------------------------------------------------------------------------------------------------
### Assign cells by Voronoi diagram
def process_nucleus_wrapper(args):
    """
    Build a convex hull polygon for a given nucleus ID if it has enough points.
    Returns the nucleus ID, polygon WKB, and centroid WKB.
    """
    nid, points = args
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            poly = Polygon(points[hull.vertices])
            return nid, (wkb.dumps(poly), wkb.dumps(poly.centroid))
        except:
            return None
    return None

def assign_points_batch(points_batch, region_map_wkb, polygons_wkb, max_dist):
    """
    Assign unlabelled points to nearest valid nucleus region
    using centroid-based KDTree lookup and shapely geometry checks.
    """
    points = np.frombuffer(points_batch, dtype=np.float64).reshape(-1, 2)
    results = np.full(len(points), np.nan, dtype=object)

    region_map = {nid: wkb.loads(wkb_data) for nid, wkb_data in region_map_wkb.items()}
    polygons = {nid: wkb.loads(wkb_data) for nid, wkb_data in polygons_wkb.items()}
    centroids = {nid: poly.centroid for nid, poly in polygons.items()}

    centroid_coords = np.array([(c.x, c.y) for c in centroids.values()])
    if len(centroid_coords) > 0:
        tree = cKDTree(centroid_coords)

        for i, pt in enumerate(points):
            point = Point(pt)
            dists, idxs = tree.query([pt], k=min(3, len(centroids)), distance_upper_bound=max_dist * 1.5)
            for dist, nid_idx in zip(dists[0], idxs[0]):
                if np.isinf(dist):
                    continue
                nid = list(centroids.keys())[nid_idx]
                if (
                    nid in region_map and nid in polygons and
                    region_map[nid].contains(point) and
                    point.distance(centroids[nid]) <= max_dist and
                    not polygons[nid].contains(point)
                ):
                    results[i] = nid
                    break
    return results

def build_region_worker(i, centroids_keys, vor_vertices, vor_regions, vor_point_region, boundary_wkb):
    """
    Build a bounded Voronoi region polygon for a given centroid index.
    Returns the polygon WKB if valid.
    """
    boundary = wkb.loads(boundary_wkb)
    nid = centroids_keys[i]
    region_idx = vor_point_region[i]
    region = vor_regions[region_idx]
    if not region or -1 in region:
        return None
    poly_points = vor_vertices[region]
    if len(poly_points) >= 3:
        poly = Polygon(poly_points).intersection(boundary)
        if not poly.is_empty and poly.is_valid:
            return nid, wkb.dumps(poly)
    return None

def process_group(name_group):
    """
    Add a suffix to each cell ID based on the hash of its cluster label,
    to distinguish cells from different clusters within the same Voronoi region.
    """
    (cell_id, cluster_id), group = name_group
    suffix = chr(65 + (hash(cluster_id) % 26))
    return pd.Series(f"{cell_id}-{suffix}", index=group.index)

def assign_cell_by_voronoi(
        df: pd.DataFrame,
        x: str = "x_location",
        y: str = "y_location",
        nucleus_label: str = "nucleus_id",
        cluster_label: str = "cluster_0.45",
        output_col: str = "updated_cell_id",
        n_workers: int = None,
        max_distance: float = None,
        dist_factor: float = 2.0,
        batch_size: int = 1000,
        verbose: bool = True
) -> pd.DataFrame:
    """Assigns transcript points to cells using Voronoi tessellation based on nucleus centroids.

    This function constructs Voronoi regions from nucleus centroids, optionally segmented 
    by cluster, and assigns each transcript to the nearest region. It supports parallel 
    processing and allows distance thresholding to restrict assignments.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing spatial transcript coordinates and nucleus assignments.
        x (str, optional): 
            Column name for x-coordinate. Defaults to "x_location".
        y (str, optional): 
            Column name for y-coordinate. Defaults to "y_location".
        nucleus_label (str, optional): 
            Column name for nucleus IDs (used as Voronoi seeds). Defaults to "nucleus_id".
        cluster_label (str, optional): 
            Column name for cluster/subtype labels to split Voronoi computation. Defaults to "cluster_0.45".
        n_workers (int or None, optional): 
            Number of parallel workers to use. If None, up to 16 cores are used. Defaults to None.
        max_distance (float or None, optional): 
            Maximum distance allowed from a centroid to assign a point. If None, will be estimated. Defaults to None.
        dist_factor (float, optional): 
            Factor to multiply the estimated max distance. Defaults to 2.0.
        batch_size (int, optional): 
            Number of unassigned points to process per batch. Defaults to 1000.
        verbose (bool, optional): 
            Whether to print progress messages during assignment. Defaults to True.

    Returns:
        pd.DataFrame: 
            Modified DataFrame with two new columns:
            - 'voronoi_assigned': Boolean indicating if a point was successfully assigned.
            - 'updated_cell_id': ID of the assigned cell after Voronoi segmentation.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 16)

    df = df.copy()
    df['voronoi_assigned'] = df[nucleus_label]

    # Step 1: Build convex hull polygons per nucleus
    unique_nuclei = df[nucleus_label].dropna().unique()
    nucleus_groups = [
        (nid, df.loc[df[nucleus_label] == nid, [x, y]].values)
        for nid in unique_nuclei
        if len(df[df[nucleus_label] == nid]) >= 3
    ]

    with mp.Pool(n_workers) as pool:
        results = pool.map(process_nucleus_wrapper, nucleus_groups,
                           chunksize=max(1, len(nucleus_groups) // (n_workers * 4)))

    valid_results = {res[0]: res[1] for res in results if res is not None}

    if len(valid_results) < 2:
        if verbose:
            print("Not enough valid nuclei found.")
        df['voronoi_assigned'] = df[nucleus_label].apply(lambda x: f"Cell-{int(x)}" if pd.notnull(x) else x)
        return df

    if verbose:
        print("Building Voronoi diagram...")

    polygons_wkb = {nid: data[0] for nid, data in valid_results.items()}
    centroids = {nid: wkb.loads(data[1]) for nid, data in valid_results.items()}

    # Estimate max assignment distance if not given
    if max_distance is None:
        areas = [wkb.loads(poly_wkb).area for poly_wkb in polygons_wkb.values()]
        max_distance = dist_factor * np.sqrt(np.median(areas) / np.pi)
        if verbose:
            print(f"Using auto-calculated max_distance: {max_distance:.2f}")

    vor = Voronoi(np.array([(c.x, c.y) for c in centroids.values()]))

    # Define Voronoi boundary based on overall data extent
    minx, miny = df[[x, y]].min()
    maxx, maxy = df[[x, y]].max()
    boundary = box(minx, miny, maxx, maxy)
    boundary_wkb = wkb.dumps(boundary)

    centroids_keys = list(centroids.keys())
    vor_vertices = vor.vertices
    vor_regions = vor.regions
    vor_point_region = vor.point_region

    # Build bounded Voronoi polygons in parallel
    with mp.Pool(n_workers) as pool:
        region_results = pool.starmap(
            build_region_worker,
            [(i, centroids_keys, vor_vertices, vor_regions, vor_point_region, boundary_wkb)
             for i in range(len(centroids))],
            chunksize=max(1, len(centroids) // (n_workers * 4))
        )

    region_map_wkb = {r[0]: r[1] for r in region_results if r is not None}

    # Only retain nuclei with complete polygon and centroid info
    valid_nids = set(region_map_wkb) & set(polygons_wkb) & set(centroids)
    region_map_wkb = {nid: region_map_wkb[nid] for nid in valid_nids}
    polygons_wkb = {nid: polygons_wkb[nid] for nid in valid_nids}
    centroids = {nid: centroids[nid] for nid in valid_nids}

    # Step 2: Assign unassigned points to closest valid region
    unassigned = df[df[nucleus_label].isna()]
    if not unassigned.empty:
        if verbose:
            print(f"Assigning {len(unassigned)} unassigned transcripts...")

        points = unassigned[[x, y]].values
        n_batches = math.ceil(len(points) / batch_size)
        batches = [points[i * batch_size:(i + 1) * batch_size].tobytes()
                   for i in range(n_batches)]

        with mp.Pool(n_workers) as pool:
            results = pool.starmap(
                assign_points_batch,
                [(b, region_map_wkb, polygons_wkb, max_distance) for b in batches],
                chunksize=max(1, n_batches // (n_workers * 4))
            )

        assignments = np.concatenate([r for r in results])
        df.loc[unassigned.index, 'voronoi_assigned'] = assignments

    # Step 3: Split each Voronoi region based on clustering
    if verbose:
        print("Splitting clusters...")

    df['voronoi_assigned'] = df['voronoi_assigned'].apply(
        lambda x: f"Cell-{int(x)}" if pd.notnull(x) else x
    )

    # groups = [(name, group) for name, group in df.groupby(['voronoi_assigned', cluster_label])]
    groups = [(name, group) for name, group in df.groupby(['voronoi_assigned', cluster_label], observed=True)]

    with mp.Pool(n_workers) as pool:
        id_series = pool.map(process_group, groups)

    df[output_col] = pd.concat(id_series)

    # Step 4: Filter spatially abnormal cells (too isolated)
    if verbose:
        print("Filtering abnormal cells...")

    if len(df) > 0:
        cell_stats = df.groupby(output_col)[[x, y]].agg(['count', 'mean'])
        if len(cell_stats) > 1:
            kdtree = cKDTree(cell_stats[[(x, 'mean'), (y, 'mean')]].values)
            distances, _ = kdtree.query(cell_stats[[(x, 'mean'), (y, 'mean')]].values, k=2)
            neighbor_dists = distances[:, 1]
            median_dist = np.median(neighbor_dists)
            valid_cells = cell_stats[neighbor_dists < 3 * median_dist].index
            df = df[df[output_col].isin(valid_cells)]

    if verbose:
        print("Assignment completed.")

    return df


# # Merge small cells that were over-segmented due to bin label constraints
def merge_small_cells(
    df: pd.DataFrame,
    x: str = 'x_location',
    y: str = 'y_location',
    cluster: str = 'cluster_col_by_bin_labels',
    cell_id: str = 'updated_cell_id',
    output: str = 'updated_cell_id_merged',
    min_UMI: int = 3,
    filter: bool = True,
    k_neighbors: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """Merges small transcriptomic cells into neighboring larger ones within the same cluster.

    Cells with total transcript (UMI) counts below a specified threshold are considered 
    "small" and are merged into the nearest neighbor cell (within the same cluster) 
    based on spatial coordinates.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing transcript-level spatial data and cell assignments.
        x (str, optional): 
            Column name for x-coordinates. Defaults to 'x_location'.
        y (str, optional): 
            Column name for y-coordinates. Defaults to 'y_location'.
        cluster (str, optional): 
            Column name indicating cluster labels to constrain merging within clusters. Defaults to 'cluster_0.45'.
        cell_id (str, optional): 
            Column name for current cell ID assignments. Defaults to 'updated_cell_id'.
        output (str, optional): 
            Column name for storing the merged cell IDs. Defaults to 'updated_cell_id_merged'.
        min_UMI (int, optional): 
            Minimum number of UMIs required for a cell to be retained as-is. Defaults to 3.
        filter (bool, optional): 
            If True, removes small cells that cannot be merged. Defaults to True.
        k_neighbors (int, optional): 
            Number of nearest neighbors to consider for merging targets. Defaults to 1.
        verbose (bool, optional): 
            If True, prints status and statistics during merging. Defaults to True.

    Returns:
        pd.DataFrame: 
            Modified DataFrame with an additional column containing merged cell IDs. 
            If `filter=True`, small unmerged cells are removed.
    """
    df = df.reset_index(drop=True).copy()

    # Precompute UMI counts and small cells
    umi_counts = df[cell_id].value_counts()
    small_cells = umi_counts[umi_counts <= min_UMI].index.to_numpy()
    if verbose:
        print(f"Found {len(small_cells)} small cells with <= {min_UMI} UMIs")

    # cell centroids
    centroids = df.groupby(cell_id)[[x, y]].mean()
    centroid_ids = centroids.index.to_numpy()
    centroid_array = centroids.to_numpy()

    # Build index lookup
    id_to_index = {cid: i for i, cid in enumerate(centroid_ids)}
    index_to_id = {i: cid for i, cid in enumerate(centroid_ids)}
    small_inds = [id_to_index[cid] for cid in small_cells]

    # KDTree query all small cell centers
    tree = KDTree(centroid_array)
    small_coords = centroid_array[small_inds]
    _, neighbors = tree.query(small_coords, k=k_neighbors + 1)

    # Create mapping from small_id to neighbor_id and cluster
    merge_map = {}
    cluster_map = df.drop_duplicates(cell_id).set_index(cell_id)[cluster].to_dict()
    small_clusters = {cid: cluster_map[cid] for cid in small_cells}

    for i, small_index in enumerate(small_inds):
        small_id = index_to_id[small_index]
        small_cluster = small_clusters[small_id]
        for ni in neighbors[i][1:]:  # skip self
            neighbor_id = index_to_id[ni]
            if neighbor_id not in small_cells and cluster_map[neighbor_id] == small_cluster:
                merge_map[small_id] = neighbor_id
                break

    # Report merge count
    df[output] = df[cell_id].map(merge_map).fillna(df[cell_id])
    if verbose:
        merged_count = sum(cid in merge_map for cid in small_cells)
        print(f"Merged {merged_count} cells out of {len(small_cells)} small cells")

    # Filter cells with less than min_UMI
    if filter:
        before_filtering = len(df[output].unique())
        df = df[df[cell_id].map(umi_counts) >= min_UMI]

        if verbose:
            filtered_count = before_filtering - len(df[output].unique())
            print(f"Filtered out {filtered_count} cells with less than {min_UMI} UMIs")

    return df


# Detect abnormal cells based on irregular shape or unusual transcript density  
def abnormal_cell_detection(
    df,
    cell_col="cell_id",
    gene_col="feature_name",
    x="x_location",
    y="y_location",
    polarity_thresh=0.6,
    density_thresh=0.75,
    umi_quantile=0.8,
    min_UMIs=5
):
    """
    Detect abnormal cells based on spatial shape irregularity and transcript density dispersion.

    This function identifies potentially abnormal or fragmented cells from spatial point data 
    (e.g., transcripts, spots, or molecules) by computing:
      - Shape polarity: a measure of elongation based on the covariance matrix of coordinates.
      - Density dispersion: standard deviation of distances from cell centroid.
    
    Cells with high elongation (high polarity) and high dispersion, or very low UMI counts, 
    are flagged as abnormal.

    Args:
        df (pd.DataFrame): Input data containing spatial coordinates and feature annotations.
        cell_col (str): Column name indicating cell/group identity (e.g., 'cell_id').
        gene_col (str): Column representing individual features (e.g., gene names or molecule IDs).
        x (str): Column name for x-coordinate.
        y (str): Column name for y-coordinate.
        polarity_thresh (float): Threshold for shape polarity to flag abnormality.
        density_thresh (float): Threshold for dispersion (std of radial distances) to flag abnormality.
        umi_quantile (float): Quantile threshold for filtering out cells with unusually high UMI counts.
        min_UMIs (int): Minimum number of UMIs required to consider polarity/density calculations valid.

    Returns:
        pd.DataFrame: A table with one row per cell, including:
            - polarity: elongation score of the cell's shape.
            - density_std: standard deviation of distance to centroid.
            - abnormal: binary flag (1 = abnormal, 0 = normal).
    """

    df = df.copy()
    df["UMI_counts"] = df.groupby(cell_col)[gene_col].transform("count")
    df_filtered = df[df["UMI_counts"] < df["UMI_counts"].quantile(umi_quantile)]

    polarity_dict = {}
    std_dict = {}
    abnormal_dict = {}

    for cell_id, group in df_filtered.groupby(cell_col):
        coords = group[[x, y]].values

        if len(coords) <= min_UMIs: 
            polarity_dict[cell_id] = np.nan
            std_dict[cell_id] = np.nan
            abnormal_dict[cell_id] = 1 
            continue

        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        polarity = np.sqrt(1 - eigvals[1] / eigvals[0]) if eigvals[0] > 0 else 0
        std_radius = np.std(np.linalg.norm(coords - coords.mean(axis=0), axis=1))

        polarity_dict[cell_id] = polarity
        std_dict[cell_id] = std_radius

        is_abnormal = ((polarity > polarity_thresh) and (std_radius > density_thresh)) or np.isnan(polarity)
        abnormal_dict[cell_id] = int(is_abnormal)

    # Merge results
    all_cells = pd.DataFrame(df[cell_col].drop_duplicates(), columns=[cell_col])
    all_cells["polarity"] = all_cells[cell_col].map(polarity_dict)
    all_cells["density_std"] = all_cells[cell_col].map(std_dict)
    all_cells["abnormal"] = all_cells[cell_col].map(abnormal_dict).fillna(0).astype(int)

    return all_cells


# Remove overlapping cells with conflicting labels at shared spatial locations
def remove_overlapping_cells(
    df, 
    cell_label="updated_cell_id", 
    x="x_location", 
    y="y_location", 
    alpha_val=0.1,
    area_percentile=5,
    containment_threshold=0.95,
    max_candidates=5,
    n_jobs=4,
    verbose=True
):
    """Removes small cells that are mostly contained within larger cells.

    This function identifies spatially small cells based on area percentile and removes 
    them if they are largely enclosed by neighboring larger cells, using alpha shapes 
    for cell boundary estimation.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing spatial coordinates and cell labels.
        cell_label (str, optional): 
            Column name for cell identifiers. Defaults to "updated_cell_id".
        x (str, optional): 
            Column name for x-coordinates. Defaults to "x_location".
        y (str, optional): 
            Column name for y-coordinates. Defaults to "y_location".
        alpha_val (float, optional): 
            Alpha parameter used in alpha shape generation for cell boundary. Defaults to 0.1.
        area_percentile (float, optional): 
            Percentile threshold to distinguish small cells. Defaults to 5.
        containment_threshold (float, optional): 
            Minimum fractional overlap required to mark a small cell as contained. Defaults to 0.95.
        max_candidates (int, optional): 
            Maximum number of nearby large cells to check for containment. Defaults to 5.
        n_jobs (int, optional): 
            Number of parallel jobs to use for processing. Defaults to 4.
        verbose (bool, optional): 
            If True, prints progress information. Defaults to True.

    Returns:
        pd.DataFrame: 
            Original DataFrame with an additional column `'keep'`, where `False` indicates 
            cells that are considered overlapping and can be removed.
    """
    if verbose:
        print("[INFO] Building cell polygons...")

    # Step 1: Group points by cell ID and filter invalid entries
    valid_df = df.dropna(subset=[cell_label, x, y])
    grouped = valid_df.groupby(cell_label)[[x, y]]
    cell_points = {label: group.values for label, group in grouped}

    # Step 2: Generate polygons for each cell using alpha shapes
    def generate_polygon(label, points):
        """Generate a polygon from a set of points using alpha shapes"""
        if len(points) < 3: return None  # Need at least 3 points for a polygon
        try:
            shape = alphashape(points, alpha_val)
            if not shape or shape.is_empty: return None
            if isinstance(shape, Polygon):
                return (label, shape, shape.area) if shape.is_valid else None
            elif isinstance(shape, MultiPolygon):
                largest = max(shape.geoms, key=lambda p: p.area)
                return (label, largest, largest.area) if largest.is_valid else None
        except Exception as e:
            if verbose: print(f"[WARN] Polygon failed for cell {label}: {str(e)[:100]}...")
            return None

    # Process polygons in parallel with progress bar if verbose
    items = list(cell_points.items())
    if verbose: items = tqdm(items, desc="Generating polygons")
    results = Parallel(n_jobs=n_jobs)(delayed(generate_polygon)(*item) for item in items)

    # Create dataframe of valid polygons with their areas
    polygons_df = pd.DataFrame(
        [r for r in results if r is not None],
        columns=[cell_label, 'polygon', 'area']
    )

    if verbose:
        print(f"[INFO] Created {len(polygons_df)} valid polygons")

    # Step 3: Classify cells as small or large based on area percentile
    area_thresh = np.percentile(polygons_df['area'], area_percentile)
    polygons_df['centroid'] = polygons_df['polygon'].apply(lambda p: (p.centroid.x, p.centroid.y))
    small_cells = polygons_df[polygons_df['area'] <= area_thresh].copy()
    large_cells = polygons_df[polygons_df['area'] > area_thresh].copy()

    if verbose:
        print(f"[INFO] Area threshold: {area_thresh:.2f}")
        print(f"[INFO] Small cells: {len(small_cells)}, Large cells: {len(large_cells)}")

    # Step 4: Build KDTree for efficient spatial queries of large cell centroids
    tree = cKDTree(np.array(large_cells['centroid'].tolist()))

    # Step 5: Check if small cells are mostly contained within large cells
    def check_overlap(row):
        """Check if a small cell is mostly contained within any nearby large cell"""
        # Query nearest large cells
        _, indices = tree.query(row['centroid'], k=min(max_candidates, len(large_cells)))
        for idx in np.atleast_1d(indices):
            try:
                # Calculate overlap area
                overlap_area = row['polygon'].intersection(large_cells.iloc[idx]['polygon']).area
                # Check if overlap exceeds containment threshold
                if overlap_area / row['area'] >= containment_threshold:
                    return {cell_label: row[cell_label], 'keep': False}
            except:
                continue
        return {cell_label: row[cell_label], 'keep': True}

    # Process overlap checks in parallel
    small_iter = tqdm(small_cells.iterrows(), total=len(small_cells), desc="Checking overlap") if verbose else small_cells.iterrows()
    overlap_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(check_overlap)(row) for _, row in small_iter
    )

    # Step 6: Merge results and mark cells to keep/remove
    polygons_df = polygons_df.merge(pd.DataFrame(overlap_results), on=cell_label, how="left")
    polygons_df["keep"] = polygons_df["keep"].fillna(True)  # Default to keeping cells without overlap checks

    # Merge results back to original dataframe while preserving original columns
    columns_to_add = ['area', 'keep']
    df_clean = df.drop(columns=[col for col in columns_to_add if col in df.columns])
    final_df = df_clean.merge(polygons_df[[cell_label] + columns_to_add], on=cell_label, how="left")

    if verbose:
        print(f"[INFO] Cells could be removed: {len(polygons_df[polygons_df['keep'] == False])}")

    return final_df


# Build cell-by-gene expression matrix and cell centroid matrix after transcript labeling and cell ID assignment
def build_cell_matrices(
    df: pd.DataFrame, 
    gene_col: str = 'gene', 
    cell_col: str = 'cell_id', 
    cluster_col: str = 'cluster',
    x_col: str = 'x', 
    y_col: str = 'y'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs a binary cell-by-gene expression matrix and a centroid coordinate matrix from transcript-level input data.

    This function processes a per-transcript DataFrame to:
    1. Generate a binary cell-by-gene expression matrix indicating whether each gene is present in each cell.
    2. Compute per-cell centroid coordinates based on the average x and y positions of its transcripts.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing per-transcript data with at least gene, cell, and spatial coordinate columns.
        gene_col (str, optional): 
            Name of the column representing gene identity. Default is 'gene'.
        cell_col (str, optional): 
            Name of the column representing cell identity. Default is 'cell_id'.
        cluster_col (str, optional): 
            Name of the column representing cluster identity. Default is 'cluster'.
        x_col (str, optional): 
            Name of the column representing x-coordinate of each transcript. Default is 'x'.
        y_col (str, optional): 
            Name of the column representing y-coordinate of each transcript. Default is 'y'.

    Returns:
        expression_matrix (pd.DataFrame): 
            A sparse binary DataFrame where rows are cells, columns are genes, and values indicate gene presence in each cell.
        metadata (pd.DataFrame): 
            A DataFrame indexed by cell ID with columns 'centroid_x', 'centroid_y', and 'cluster'
            representing the average spatial position of transcripts assigned to each cell and the cluster identity.
    """
    # Drop rows with missing required fields
    df = df.dropna(subset=[gene_col, cell_col, cluster_col, x_col, y_col])
    
    # Convert to categorical for efficient indexing
    df[cell_col] = df[cell_col].astype('category')
    df[gene_col] = df[gene_col].astype('category')
    df[cluster_col] = df[cluster_col].astype('category')

    # Get integer codes for cells and genes
    cell_categories = df[cell_col].cat.categories
    gene_categories = df[gene_col].cat.categories
    cell_index = df[cell_col].cat.codes.to_numpy()
    gene_index = df[gene_col].cat.codes.to_numpy()

    # Build sparse binary matrix
    values = np.ones(len(df), dtype=int)
    expression_sparse = coo_matrix(
        (values, (cell_index, gene_index)),
        shape=(len(cell_categories), len(gene_categories))
    )

    expression_matrix = pd.DataFrame.sparse.from_spmatrix(
        expression_sparse, 
        index=cell_categories, 
        columns=gene_categories
    )

    # Compute per-cell spatial centroid
    metadata = (
        df.groupby(cell_col, observed=True)[[x_col, y_col]]
        .mean()
        .rename(columns={x_col: 'centroid_x', y_col: 'centroid_y'})
    )

    # Assign the most frequent cluster per cell
    metadata['cluster'] = df.groupby(cell_col, observed=True)[cluster_col].agg(lambda x: x.mode().iloc[0])
    metadata.index.name = None

    return expression_matrix, metadata


### Plot segmented cells
def plot_segmented_cells(
    df: pd.DataFrame, 
    cell_label: str = "cell_label", 
    cluster_label: str = "binned_cluster", 
    x: str = "x_location", 
    y: str = "y_location", 
    fig_title: str = "Segmented Cells", 
    h: float = 5, 
    w: float = 5, 
    cmap: Optional[Colormap] = None,
    alpha_val: float = 0.1, 
    linewidth: float = 1.5, 
    s: float = 20, 
    alpha: float = 0.9
) -> None:
    """Plots segmented cell boundaries using alpha shapes and colors points by cluster.

    This function generates a scatter plot of transcripts colored by cluster labels 
    and overlays cell boundaries computed from alpha shapes of each cell.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing per-transcript spatial coordinates and cell/cluster labels.
        cell_label (str, optional): 
            Column name representing per-transcript cell assignment. Defaults to "cell_label".
        cluster_label (str, optional): 
            Column name for cluster/group assignment used for coloring points. Defaults to "binned_cluster".
        x (str, optional): 
            Column name for x-coordinate. Defaults to "x_location".
        y (str, optional): 
            Column name for y-coordinate. Defaults to "y_location".
        fig_title (str, optional): 
            Title to be shown on the plot. Defaults to "Segmented Cells".
        h (float, optional): 
            Height of the figure in inches. Defaults to 5.
        w (float, optional): 
            Width of the figure in inches. Defaults to 5.
        cmap (Optional[Colormap], optional): 
            Colormap for cluster coloring. If None, the default matplotlib color cycle is used. Defaults to None.
        alpha_val (float, optional): 
            Alpha parameter used to compute alpha shapes for boundaries. Defaults to 0.1.
        linewidth (float, optional): 
            Width of boundary lines for each cell. Defaults to 1.5.
        s (float, optional): 
            Size of scatter points. Defaults to 20.
        alpha (float, optional): 
            Transparency of scatter points. Defaults to 0.9.

    Returns:
        None: 
            Displays the plot inline; does not return anything.
    """
    _, ax = plt.subplots(figsize=(w, h))

    # Draw cell boundaries
    grouped = df.dropna(subset=[cell_label]).groupby(cell_label)
    for label, group in grouped:
        points = group[[x, y]].to_numpy()
        if len(points) < 3:
            continue
        shape = alphashape(points, alpha=alpha_val)
        if isinstance(shape, MultiPolygon):
            for poly in shape.geoms:
                ax.plot(*poly.exterior.xy, 'k-', linewidth=linewidth, alpha=0.6)
        elif isinstance(shape, Polygon):
            ax.plot(*shape.exterior.xy, 'k-', linewidth=linewidth, alpha=0.6)

    # Colormap setup
    unique_clusters = df[cluster_label].dropna().unique()
    num_clusters = len(unique_clusters)
    if cmap is None:
        cmap = plt.get_cmap("Set1", num_clusters)
    
    # Map cluster values to integers (for color)
    cluster_to_int = {k: i for i, k in enumerate(sorted(unique_clusters))}
    cluster_colors = df[cluster_label].map(cluster_to_int)

    # Plot all points
    ax.scatter(df[x], df[y], s=s, alpha=alpha, c=cluster_colors, cmap=cmap, edgecolors='none')

    # Plot style
    ax.set_title(fig_title)
    ax.axis('off')
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show() 