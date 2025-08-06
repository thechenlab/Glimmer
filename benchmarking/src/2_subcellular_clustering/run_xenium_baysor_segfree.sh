#!/bin/bash

### Run Baysor on the xenium_test dataset
file_path="/data/qiyu/spatialRegion/benchmark/Public_data/Xenium_subsets/"
baysor="/home/qiyu/.julia/bin/baysor"
range_list=(10-11  13-14  15-16  17-18  21-22  23-24  27-28  29-30  31-32  33-34  39-40  41-42  43-44  7-8  9-10)

cd $file_path
for range in "${range_list[@]}"
do
    echo "-----------------------------------"
    echo "Processing range $range"
    cd ${file_path}${range}

    # Clean transcripts.tsv to transcripts.csv
    gunzip -c transcripts.tsv.gz > transcripts.tsv
    awk -v OFS='\t' 'NR==1 {print $0, "Z"; next} {print $0, "1"}' transcripts.tsv > transcripts_z.tsv
    sed 's/\t/,/g' transcripts_z.tsv > transcripts.csv
    rm transcripts.tsv transcripts_z.tsv
    awk -F',' '
        NR==1 || ($3 !~ /^"?NegControl/ && $3 !~ /^"?BLANK_/)
        ' transcripts.csv > filtered.csv
    mv filtered.csv transcripts.csv
    
    # Run Baysor
    output="output_baysor_k15"
    [ -d $output ] && rm -rf $output
    mkdir -p $output

    # Run Baysor with minimal umis at 15, suggested by Xenium
    # https://www.10xgenomics.com/analysis-guides/using-baysor-to-perform-xenium-cell-segmentation
    $baysor segfree "transcripts.csv" -x X -y Y -z Z -g gene -k 15 -m 15 -o $output/ncv_results.loom

    echo "Done processing range $range"
    echo "-----------------------------------"
    echo
done

# gunzip -c transcripts.tsv.gz > transcripts.tsv
# awk -v OFS='\t' 'NR==1 {print $0, "Z"; next} {print $0, "1"}' transcripts.tsv > transcripts_with_z.tsv
# awk -v OFS='\t' 'NR==1 {print $1, $2, "Z", $3, $4; next} {print $1, $2, "1", $3, $4}' transcripts.tsv > transcripts_with_z.tsv
# mv transcripts_with_z.tsv transcripts.tsv
# sed 's/\t/,/g' transcripts.tsv > transcripts.csv
# rm transcripts.tsv
# # awk 'BEGIN {FS="\t"; OFS=","} {$1=$1}1' transcripts.tsv > transcripts.csv
# /home/qiyu/.julia/bin/baysor segfree transcripts.csv -k 5 -x X -y Y -z Z -g gene -m 10 -o ncv_results.loom
