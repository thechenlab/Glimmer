#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=:16:8

# Usage:
#   ./run_region_benchmarking.sh <data_type> <model_idx> <device_num>
#
# Arguments:
#   data_type   - Type of dataset to use. Must be one of: slide_seq, merfish
#   model_idx   - Index of model to run. Must be one of:
#                   0 - Glimmer
#                   1 - GraphST
#                   2 - SPIN
#                   3 - STAGATE
#                   4 - SpaceFlow
#                   5 - SCANIT
#                   6 - Banksy
#   device_num  - GPU device index. Must be one of: 0, 1, 2, 3
#
# Examples:
#   ./run_region_benchmarking.sh slide_seq 0 0     # Run Glimmer on slide_seq using GPU 0
#   ./run_region_benchmarking.sh merfish 4 1       # Run SpaceFlow on merfish using GPU 1


data_type=$1
model_idx=$2
device_num=$3

# parse input data_type
if [ "$data_type" != "slide_seq" ] && [ "$data_type" != "merfish" ]; then
    echo "data_type must be one of slide_seq and merfish"
    exit 1
fi

# parse input model name
if [ "$model_idx" == "0" ]; then
    echo "Running Glimmer"
elif [ "$model_idx" == "1" ]; then
    echo "Running GraphST"
elif [ "$model_idx" == "2" ]; then
    echo "Running SPIN"
elif [ "$model_idx" == "3" ]; then
    echo "Running STAGATE"
elif [ "$model_idx" == "4" ]; then
    echo "Running SpaceFlow"
elif [ "$model_idx" == "5" ]; then
    echo "Running SCANIT"
elif [ "$model_idx" == "6" ]; then
    echo "Running Banksy"
else
    echo "model_idx must be one of 0, 1, 2, 3, 4, 5 and 6"
    exit 1
fi

# parse input device number
if [ "$device_num" != "0" ] && [ "$device_num" != "1" ] && [ "$device_num" != "2" ] && [ "$device_num" != "3" ]; then
    echo "device_num must be one of 0, 1, 2 and 3"
    exit 1
fi

# parse input data name
if [ "$data_type" == "slide_seq" ]; then
    echo "Processing slide-seq data"
    data_names=("01" "02" "03" "04" "05")
elif [ "$data_type" == "merfish" ]; then
    echo "Processing merfish data"
    data_names=(0 1 2 3 4)
fi  

# run model
python_path="/home/qiyu/miniconda3/envs/py310/bin/python"
R_path="/home/qiyu/miniconda3/envs/py310/bin/Rscript"
src_path="./src"

seed_list=(1 2 3 4 5)
device="cuda:$device_num"
for name in "${data_names[@]}"; do
    for seed in "${seed_list[@]}"; do
        echo "Processing data $data_type $name with seed $seed"
        if [ $model_idx == "6" ]; then
            $R_path ${src_path}/run_region_Banksy.R $name $seed $data_type
        else
            $python_path ${src_path}/run_region_methods.py $model_idx $name $seed $data_type $device
        fi
    done
done
