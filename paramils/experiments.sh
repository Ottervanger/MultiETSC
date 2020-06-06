#!/bin/bash

set -e
# change dir to script location
cd "${0%/*}"

UCR="/vol/share/groups/liacs/scratch/UCR/"
DATASET="ECG200"

# produce train-validate split from training data
INSTANCE=$(python validationsplitter.py --folds=5 $UCR$DATASET"/"$DATASET"_TRAIN.tsv")
echo $INSTANCE
perl -pi -e "s|^instance_file =.*$|${INSTANCE}|g" "earlytsc/scenario.txt"
echo $UCR$DATASET"/"$DATASET"_TRAIN.tsv" $UCR$DATASET"/"$DATASET"_TEST.tsv" > earlytsc/test_list.txt

./moparamils --scenario-file earlytsc/scenario.txt

# get output file
OUTFILE=$(ls -Art output/earlytsc/detailed-traj-run-*.csv | tail -n 1)
# extract the Pareto front
grep -E "\[[0-9e. ]*,[0-9e. ]*\]" "$OUTFILE" | cut -d"\"" -f4 | perl -pe "s/[\[\] ]//g"

