import math
import xml.etree.ElementTree as ET
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, Voronoi
from scipy.spatial import KDTree
import zarr
from zarr import open as open_zarr
from tifffile import TiffFile
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely import wkb
from alphashape import alphashape
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional



# ------------------------------------------------------------------------------------------------
# When using DAPI image to segment nuclei, we need to get the pixel size from the OME-XML file
# ------------------------------------------------------------------------------------------------
def get_pixel_size_from_ome(xml_string):
    root = ET.fromstring(xml_string)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    pixels = root.find('.//ome:Pixels', namespaces=ns)
    px_size_x = float(pixels.attrib['PhysicalSizeX'])
    px_size_y = float(pixels.attrib['PhysicalSizeY'])
    return px_size_x, px_size_y

def open_zarr(path):
    store = zarr.ZipStore(path, mode='r') if path.endswith(".zip") else zarr.DirectoryStore(path)
    return zarr.group(store=store)



# ------------------------------------------------------------------------------------------------
# assign nucleus ids from DAPI image to transcripts
# ------------------------------------------------------------------------------------------------
def assign_nucleus_ids_to_transcripts(
        transcripts_path: str, 
        cells_path: str, 
        trans_df_save_file: str, 
        pixel_size_x_um: float = 0.2125, 
        pixel_size_y_um: float = 0.2125
) -> pd.DataFrame:
    """
    Assign nucleus IDs to each transcript based on spatial coordinates and a segmentation mask.

    Parameters
    ----------
    transcripts_path : str
        Path to the `transcripts.csv.gz` file.
    cells_path : str
        Path to the `cells.zarr.zip` file.
    trans_df_save_file : str
        File path to save the updated transcript dataframe.
    pixel_size_x_um : float
        Microns per pixel in x-direction (default 0.2125).
    pixel_size_y_um : float
        Microns per pixel in y-direction (default 0.2125).

    Returns
    -------
    pd.DataFrame
        Transcript dataframe with nucleus IDs.
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
# Assign cells by Voronoi diagram
# ------------------------------------------------------------------------------------------------
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
        n_workers: int = None,
        max_distance: float = None,
        batch_size: int = 1000,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Assign transcript points to cells using Voronoi regions built from nucleus centroids,
    while optionally splitting based on clustering labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with spatial point coordinates and nucleus ID assignment.
    x, y : str
        Column names for spatial coordinates.
    nucleus_label : str
        Column name for nucleus assignment.
    cluster_label : str
        Column name for clustering used to distinguish cell subgroups.
    n_workers : int or None
        Number of parallel workers to use. Default uses up to 16 cores.
    max_distance : float or None
        Maximum allowed distance from centroid to assign a point. If None, it is estimated.
    batch_size : int
        Number of unassigned points to process per batch.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'voronoi_assigned' and 'updated_cell_id' columns added.
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
        max_distance = 2 * np.sqrt(np.median(areas) / np.pi)
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

    groups = [(name, group) for name, group in df.groupby(['voronoi_assigned', cluster_label])]

    with mp.Pool(n_workers) as pool:
        id_series = pool.map(process_group, groups)

    df['updated_cell_id'] = pd.concat(id_series)

    # Step 4: Filter spatially abnormal cells (too isolated)
    if verbose:
        print("Filtering abnormal cells...")

    if len(df) > 0:
        cell_stats = df.groupby('updated_cell_id')[[x, y]].agg(['count', 'mean'])
        if len(cell_stats) > 1:
            kdtree = cKDTree(cell_stats[[(x, 'mean'), (y, 'mean')]].values)
            distances, _ = kdtree.query(cell_stats[[(x, 'mean'), (y, 'mean')]].values, k=2)
            neighbor_dists = distances[:, 1]

            median_dist = np.median(neighbor_dists)
            valid_cells = cell_stats[neighbor_dists < 3 * median_dist].index
            df = df[df['updated_cell_id'].isin(valid_cells)]

    if verbose:
        print("Assignment completed.")

    return df



# ------------------------------------------------------------------------------------------------
# Merge small cells
# ------------------------------------------------------------------------------------------------
def merge_small_cells(
    df: pd.DataFrame,
    x: str = 'x_location',
    y: str = 'y_location',
    cluster: str = 'cluster_0.45',
    cell_id: str = 'updated_cell_id',
    output: str = 'updated_cell_id_merged',
    min_UMI: int = 5,
    filter: bool = True,
    k_neighbors: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge small spatial transcriptomic cells (based on UMI counts) into nearby larger cells within the same cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with transcript-level spatial data and cell assignments.
    x : str, default='x_location'
        Column name for x-coordinates.
    y : str, default='y_location'
        Column name for y-coordinates.
    cluster : str, default='cluster_0.45'
        Column indicating the cluster assignment of each cell.
    cell_id : str, default='updated_cell_id'
        Column indicating current cell ID of each transcript.
    output : str, default='updated_cell_id_merged'
        Name of the output column for updated cell IDs after merging.
    min_UMI : int, default=5
        Minimum UMI count threshold for a cell to avoid being considered "small".
    filter : bool, default=True
        Whether to filter out small cells that could not be merged.
    k_neighbors : int, default=3
        Number of nearest neighbors to search when finding merge targets for small cells.
    verbose : bool, default=True
        Whether to print information during the merging process.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column indicating merged cell IDs. Optionally filtered to exclude unmerged small cells.
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



# ------------------------------------------------------------------------------------------------
# Remove overlapping cells
# ------------------------------------------------------------------------------------------------
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
    """
    Remove small cells that are mostly contained within larger cells.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing cell coordinates and labels
        cell_label (str): Column name for cell identifiers
        x (str): Column name for x-coordinates
        y (str): Column name for y-coordinates
        alpha_val (float): Alpha parameter for alphashape polygon generation
        area_percentile (float): Percentile threshold to define small vs large cells
        containment_threshold (float): Minimum overlap ratio to consider a small cell contained
        max_candidates (int): Maximum nearby large cells to check for overlap
        n_jobs (int): Number of parallel jobs to run
        verbose (bool): Whether to print progress information
    
    Returns:
        pd.DataFrame: Original dataframe with added 'keep' column (False for cells to remove)
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



# ------------------------------------------------------------------------------------------------
# Plot segmented cells
# ------------------------------------------------------------------------------------------------
def plot_segmented_cells(
    df: pd.DataFrame, 
    cell_label: str = "cell_label", 
    cluster_label: str = "binned_cluster", 
    x: str = "x_location", 
    y: str = "y_location", 
    fig_title: str = "Segmented Cells (ScaleFlow)", 
    h: float = 5, 
    w: float = 5, 
    cmap: Optional[Colormap] = None,
    alpha_val: float = 0.1, 
    linewidth: float = 1.5, 
    s: float = 20, 
    alpha: float = 0.9
) -> None:
    """
    Plot segmented cell boundaries using alpha shapes and color points by cluster.

    Parameters
    ----------
    df : pd.DataFrame
    cell_label : str, per-cell assignment ID
    cluster_label : str, per-transcript cluster label (for color)
    x, y : str, coordinate column names
    alpha_val : float, alpha parameter for alpha shapes
    fig_title : str, title of the plot 
    h, w : float, figure height and width
    linewidth : float, boundary line width
    s : float, point size
    alpha : float, point transparency 
    cmap : matplotlib colormap or None
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
