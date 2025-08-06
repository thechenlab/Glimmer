import warnings
import random
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import argparse
import scib_metrics as scib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numba.core.errors import NumbaDeprecationWarning
from .knn import *
from .cas import compute_cas
from .mlami import compute_mlami
from .nasw import compute_nasw

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# set up data and model
# Notes: `Glimmer` is the old name of `Glimmer`
data_type = 'slide_seq'
path = "/data/qiyu/spatialRegion/benchmark/results/"
data_names = ["01", "02", "03", "04", "05"]
model = ["Glimmer", "GraphST", "SPIN", "SpaceFlow", "SCANIT", "Banksy"]

use_rep_list = [f"{model}_emb_seed{seed}" for model in model for seed in [1, 2, 3, 4, 5]]
use_rep_list = [
    item.replace('Glimmer_', 'Our_') if 'Glimmer_' in item else item
    for item in use_rep_list
]


# parameters
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

argparser = argparse.ArgumentParser()
argparser.add_argument("-cas", type=str2bool, default=False)
argparser.add_argument("-mlami", type=str2bool, default=False)
argparser.add_argument("-nasw", type=str2bool, default=False)
argparser.add_argument("-cnmi", type=str2bool, default=False)
args = argparser.parse_args()
run_cas = args.cas
run_mlami = args.mlami
run_nasw = args.nasw
run_cnmi = args.cnmi

# set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# computing CAS for all data
if run_cas:
    cas_list = []
    for data in data_names:
        adata = sc.read(f"{path}/{data_type}/{data}/all_res.h5ad")

        for use_rep in use_rep_list:
            seed = 42
            sc.pp.neighbors(adata, use_rep=use_rep, random_state=seed)
            sc.tl.leiden(adata, resolution=1, random_state=seed)
            print(f"\nComputing CAS for {use_rep} on {data}")
            spatial_key = "spatial"
            latent_key = use_rep

            cas = compute_cas(
                adata=adata,
                cell_type_key="leiden",
                batch_key=None,
                spatial_knng_key=f"{spatial_key}_15knng",
                latent_knng_key=f"{latent_key}_15knng",
                spatial_key=spatial_key,
                latent_key=latent_key,
                seed=seed)
            
            print(f"\nCAS for {use_rep} on {data} is {cas}\n")
            cas_list.append((use_rep, cas))

    df = pd.DataFrame(cas_list, columns=['use_rep', 'CAS'])
    df['method'] = df['use_rep'].apply(lambda x: x.split('_')[0])
    df['seed'] = df['use_rep'].apply(lambda x: x.split('_')[-1])
    df['sample'] = [item for item in data_names for _ in range(len(use_rep_list))]
    df.to_csv(f"{path}/{data_type}/CAS.csv", index=False)


# computing MLAMI for all data
if run_mlami:
    mlami_list = []
    for data in data_names:
        adata = sc.read(f"{path}/{data_type}/{data}/all_res.h5ad")

        for use_rep in use_rep_list:
            print(f"Computing MLAMI for {use_rep} on {data}")
            spatial_key = "spatial"
            latent_key = use_rep
            seed = 42

            mlami = compute_mlami(
                adata=adata,
                batch_key=None,
                spatial_knng_key=f"{spatial_key}_15knng",
                latent_knng_key=f"{latent_key}_15knng",
                spatial_key=spatial_key,
                latent_key=latent_key,
                seed=seed)
            
            print(f"\nMLAMI for {use_rep} on {data} is {mlami}\n")
            mlami_list.append((use_rep, mlami))

    df = pd.DataFrame(mlami_list, columns=['use_rep', 'MLAMI'])
    df['method'] = df['use_rep'].apply(lambda x: x.split('_')[0])
    df['seed'] = df['use_rep'].apply(lambda x: x.split('_')[-1])
    df['sample'] = [item for item in data_names for _ in range(len(use_rep_list))]
    df.to_csv(f"{path}/{data_type}/MLAMI.csv", index=False)


# computing NASW for all data
if run_nasw:
    nasw_list = []
    for data in data_names:
        adata = sc.read(f"{path}/{data_type}/{data}/all_res.h5ad")

        for use_rep in use_rep_list:
            if adata.obsm[use_rep].shape[1] > 1000:
                print(f"Computing PCA for {use_rep}")
                X_standardized = StandardScaler().fit_transform(adata.obsm[use_rep])
                pca = PCA(n_components=100)
                adata.obsm[use_rep] = pca.fit_transform(X_standardized)

            print(f"Computing NASW for {use_rep} on {data}")
            seed = 42
            latent_key = use_rep

            nasw = compute_nasw(
                    adata=adata,
                    latent_knng_key=f"{latent_key}_15knng",
                    latent_key=latent_key,
                    seed=seed)
            print(f"\nNASW for {use_rep} on {data} is {nasw}\n")
            nasw_list.append((use_rep, nasw))

    df = pd.DataFrame(nasw_list, columns=['use_rep', 'NASW'])
    df['method'] = df['use_rep'].apply(lambda x: x.split('_')[0])
    df['seed'] = df['use_rep'].apply(lambda x: x.split('_')[-1])
    df['sample'] = [item for item in data_names for _ in range(len(use_rep_list))]
    df.to_csv(f"{path}/{data_type}/NASW.csv", index=False)

# computing CNMI for all data
if run_cnmi:
    cnmi_list = []
    for data in data_names:
        adata = sc.read(f"{path}/{data_type}/{data}/all_res.h5ad")

        for use_rep in use_rep_list:
            print(f"Computing CNMI for {use_rep} on {data}")
            latent_key = use_rep
            cell_type_key = "leiden"
            seed = 42

            sc.pp.neighbors(adata, use_rep=use_rep, random_state=seed)
            sc.tl.leiden(adata, resolution=1, random_state=seed)
            res = scib.nmi_ari_cluster_labels_kmeans(adata.obsm[latent_key], labels=adata.obs[cell_type_key])
            cnmi = res['nmi']

            print(f"\nCNMI for {use_rep} on {data} is {cnmi}\n")
            cnmi_list.append((use_rep, cnmi))

    df = pd.DataFrame(cnmi_list, columns=['use_rep', 'CNMI'])
    df['method'] = df['use_rep'].apply(lambda x: x.split('_')[0])
    df['seed'] = df['use_rep'].apply(lambda x: x.split('_')[-1])
    df['sample'] = [item for item in data_names for _ in range(len(use_rep_list))]
    df.to_csv(f"{path}/{data_type}/CNMI.csv", index=False)