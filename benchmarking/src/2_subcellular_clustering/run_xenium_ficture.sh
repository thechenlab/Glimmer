#!/bin/bash

file_path="/data/qiyu/spatialRegion/benchmark/Public_data/Xenium_subsets/"
ficture="/home/qiyu/miniconda3/envs/py39/bin/ficture"
range_list=(10-11  13-14  15-16  17-18  21-22  23-24  27-28  29-30  31-32  33-34  39-40  41-42  43-44  7-8  9-10)

cd $file_path
for range in "${range_list[@]}"
do
    echo -e "\nProcessing range $range"
    cd ${file_path}/${range}

    # clean transcripts.tsv.gz
    gunzip -c transcripts.tsv.gz | 
    awk -F'\t' 'NR==1 || ($3 !~ /^(NegControl|BLANK_)/)' | 
    bgzip > filtered_transcripts.tsv.gz

    # clean features.tsv.gz
    gunzip -c features.tsv.gz | 
    awk -F'\t' 'NR==1 || ($1 !~ /^(NegControl|BLANK_)/)' | 
    bgzip > filtered_features.tsv.gz

    # run ficture
    transcripts="filtered_transcripts.tsv.gz"
    features="filtered_features.tsv.gz"
    output="output_pixel_id_cleaned"
    [ -d $output ] && rm -rf $output
    mkdir -p $output
    
    echo "Running ficture on range $range"
    echo "transcripts: $transcripts"
    echo "features: $features"
    echo "output: $output"

    $ficture run_together --in-tsv $transcripts --in-feature $features --out-dir $output --all --n-factor 5,6,7
    echo "Done processing range $range"
    echo "-----------------------------------"
done

