.libPaths("/home/qiyu/miniconda3/envs/py39/lib/R/library")
suppressMessages(library(Banksy))
suppressMessages(library(rhdf5))
suppressMessages(library(scater))
suppressMessages(library(SummarizedExperiment))
suppressMessages(library(SpatialExperiment))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))
suppressMessages(library(SeuratWrappers))

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
data_name <- args[1]
seed <- as.integer(args[2])
data_type <- args[3]

main_path = "/data/qiyu/spatialRegion/benchmark/"

set.seed(seed)
save_path = sprintf("%s/%s/%s/", main_path, data_type, data_name)
if (!dir.exists(save_path)) {dir.create(save_path, recursive = TRUE)}
dataset_paths <- list(
  slide_seq = paste0(main_path, "Public_data/Brain_Slideseq/Subset_Puck_Num_"),
  merfish = paste0(main_path, "Public_data/MERFISH/"),
  visium = paste0(main_path, "Public_data/1.DLPFC/")
)

# Define common functions
load_h5ad_data <- function(file_path, data_type) {
  if (data_type == "slide_seq") {
    exprs_data <- as.numeric(h5read(file_path, "raw/X/data"))
    indices <- h5read(file_path, "raw/X/indices")
    indptr <- h5read(file_path, "raw/X/indptr")
    n_cells <- length(indptr) - 1
    n_features <- max(indices) + 1  
    exprs_matrix <- Matrix::sparseMatrix(i = indices + 1, p = indptr, x = exprs_data, dims = c(n_features, n_cells))

  } else if (data_type == "merfish") {
    exprs_matrix <- as.matrix(h5read(file_path, "X"))
    rownames(exprs_matrix) <- h5read(file_path, "var/Unnamed: 0")
    colnames(exprs_matrix) <- h5read(file_path, "obs/Unnamed: 0")
  }

  locs <- t(h5read(file_path, "obsm/spatial"))
  SpatialExperiment(assay = list(counts = exprs_matrix), spatialCoords = locs)
}

# Define functions for running Banksy
run_banksy <- function(obj, data_type, lambda, k_geom, seed) {
  obj <- computeLibraryFactors(obj)
  assay(obj, "normcounts") <- normalizeCounts(obj, log = FALSE)
  message(paste0("Starting Banksy run ", seed, " for ", data_type, "."))
  
  start_time <- Sys.time()
  obj <- Banksy::computeBanksy(obj, assay_name = "normcounts", compute_agf = TRUE, k_geom = k_geom, seed = seed)
  obj <- Banksy::runBanksyPCA(obj, use_agf = TRUE, seed = seed, lambda = lambda)
  message(paste0("Dim names: ", names(obj@int_colData@listData$reducedDims)))
  message(paste0("Banksy run ", seed, " for ", data_type, " completed in ", round(Sys.time() - start_time, 2), " seconds."))

  return(obj@int_colData@listData$reducedDims$PCA_M1_lam0.8)
}

# Load data and run Banksy
if (data_type == "visium") {
  load_visium_data <- function(data_name) {
    file_fold <- file.path(dataset_paths[['visium']], data_name)
    adata <- Load10X_Spatial(file_fold, filename = "filtered_feature_bc_matrix.h5")
    metadata <- read.csv(file.path(file_fold, "metadata.tsv"), sep = "\t")
    adata$ground_truth <- metadata$layer_guess
    adata <- adata[, !is.na(adata$ground_truth)]
    return(adata)
  }
  
  seu <- load_visium_data(data_name)
  seu <- NormalizeData(seu) %>% FindVariableFeatures() %>% ScaleData()
  seu <- RunBanksy(seu, lambda = 0.2, verbose = FALSE, assay = "Spatial", 
                   slot = "data", features = "variable", k_geom = 15)
  seu <- RunPCA(seu, assay = "BANKSY", features = rownames(seu), verbose = FALSE)
  emb <- seu@reductions$pca@cell.embeddings
  
} else {
  if (data_type == "slide_seq") {
    file_path <- paste0(dataset_paths[["slide_seq"]], data_name, ".h5ad")
  } else if (data_type == "merfish") {
    data_ids <- c(
      "MERFISH_0.04_20241109014506.h5ad", 
      "MERFISH_0.14_20241109015203.h5ad",
      "MERFISH_0.24_20241109015212.h5ad", 
      "MERFISH_0.09_20241109014907.h5ad",
      "MERFISH_0.19_20241109015208.h5ad"
    )
    data_name = as.numeric(data_name) + 1
    file_path <- file.path(paste0(dataset_paths[["merfish"]], data_ids[data_name]))
  }
  
  se <- load_h5ad_data(file_path, data_type)
  emb <- run_banksy(se, data_type, lambda = 0.8, k_geom = c(15, 30), seed = seed)
}

message(paste0("Saving Banksy embeddings"))
file_name <- paste0(save_path, "Banksy_emb_seed", seed, ".txt")
write.table(emb, file=file_name, sep=",", row.names=FALSE, col.names=FALSE)