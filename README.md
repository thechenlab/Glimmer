# Glimmer-ST

**A Unified Framework for Graph-Based Representation of Spatial Structures Across Modalities and Scales**

Glimmer-ST is a Python package designed to construct and learn spatial graphs from spatial transcriptomics data. It enables robust modeling of spatial neighborhoods across diverse platforms such as Slide-tags, Slide-seq, Xenium, MERFISH, and more. The framework supports large-scale datasets, efficient neighbor weighting, and seamless integration with popular bioinformatics libraries.

---

## ✨ Features

- Learn neighbor weights via Dirichlet energy minimization  
- Fully compatible with `AnnData` and `Scanpy` workflows  
- Scales to millions of cells using multiprocessing  
- Supports spatial domain identification and segmentation-free clustering  

---

## 📦 Installation

> Requires Python >= 3.10

Install directly from GitHub:

```bash
pip install git+https://github.com/thechenlab/Glimmer.git
```

To ensure CUDA compatibility with PyTorch (if using GPU):
```bash
pip install torch==2.7.0+cu126 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🧬 Quick Start
```python
import scanpy as sc
from glimmer_st.model import train_neighbor_weights

# Load spatial transcriptomics dataset
adata = sc.read_h5ad("example_data.h5ad")

# Learn neighbor weights based on spatial structure
adata = train_neighbor_weights(adata, spatial_key='spatial', k=50)

# Access the learned weights
weights = adata.obsp['neighbor_weights']
```

## 📚 Tutorial
The detailed tutorial can be found [here](https://thechenlab.github.io/Glimmer/).

📫 Contact
Qiyu Gong

📧 gongqiyu@broadinstitute.org & qiyugong23@gmail.com