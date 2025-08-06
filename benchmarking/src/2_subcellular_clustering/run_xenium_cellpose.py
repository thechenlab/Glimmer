import os
import numpy as np
import pandas as pd
from cellpose import models
import tifffile
import subprocess


main_path = '/data/qiyu/spatialRegion/'
output_path = main_path + "benchmark/Public_data/Xenium_subsets/"
dir_path = main_path + "data/xenium/xenium_human_non_diseased_lymph_node/"
img_path = f"{dir_path}xenium_outs/morphology_focus.ome.tif"


def load_transcripts(transcript_path):
    df = pd.read_csv(transcript_path, sep='\t', compression='gzip')
    return df

def calculate_transcript_bounds(df):
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    return (x_min, x_max, y_min, y_max)

def transcript_to_img_coords(df, resolution_um_per_pixel=0.2125):
    df['x_pixel'] = (df['X'] / resolution_um_per_pixel).astype(int)
    df['y_pixel'] = (df['Y'] / resolution_um_per_pixel).astype(int)
    return df

def save_normalized_image(fov, save_path, clip=True, clip_percentile=95, return_img=False):
    fov = fov.astype(np.float32)
    if clip:
        percentile_val = np.percentile(fov, clip_percentile)
        fov = np.clip(fov, 0, percentile_val)
        fov = fov / percentile_val
    else:
        fov = fov / fov.max()
    gray_uint8 = (fov * 255).astype(np.uint8)
    tifffile.imwrite(save_path, gray_uint8)
    if return_img:
        return gray_uint8

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

for fov_name in fov_names:
    print(f"Processing {fov_name}...")
    trans_path = f"{output_path}{fov_name}/transcripts.tsv.gz"
    out_img_path = f"{output_path}{fov_name}/Cellpose_cyto3_50"
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)

    df = load_transcripts(trans_path)
    df = transcript_to_img_coords(df)
    img = tifffile.imread(img_path, is_ome=True, level=0)
    fov, (x1, y1, x2, y2) = extract_fov(img, df)
    save_normalized_image(fov, f"{out_img_path}/fov_8bit.tif", clip=False, clip_percentile=95)

    cmd = [
        "python", "-m", "cellpose",
        "--image_path", f"{out_img_path}/fov_8bit.tif",
        "--pretrained_model", "cyto3",
        "--save_tif",
        "--verbose",
        "--chan", "0",
        "--diameter", "50",
        "--use_gpu",
        "--gpu_device", "0"
    ]
    subprocess.run(cmd, check=True)
