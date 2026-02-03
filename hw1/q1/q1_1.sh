#!/usr/bin/env bash

APRIORI=$1
FPGROWTH=$2
DATASET=$3
OUTDIR=$4

mkdir -p "$OUTDIR"


LOGFILE="$OUTDIR/runtime.csv"
rm -f "$LOGFILE"

echo "support,apriori_time,fpgrowth_time" > "$LOGFILE"

SUPPORTS=(90 50 25 10 5)

echo "Warming file cache..."
cat "$DATASET" > /dev/null

for S in "${SUPPORTS[@]}"; do

    echo "Running support = $S%"

    APR_OUT="$OUTDIR/ap$S.txt"
    FP_OUT="$OUTDIR/fp$S.txt"
    touch "$APR_OUT"
    touch "$FP_OUT"

    start=$(date +%s.%N)

    timeout 3600 "$APRIORI" -s$S "$DATASET" "$APR_OUT"
    status=$?

    end=$(date +%s.%N)

    if [ $status -eq 124 ]; then
        apr_time="3600"
    else
        apr_time=$(echo "$end - $start" | bc)
    fi

    start=$(date +%s.%N)

    "$FPGROWTH" -s$S "$DATASET" "$FP_OUT"

    end=$(date +%s.%N)
    fp_time=$(echo "$end - $start" | bc)

    echo "$S,$apr_time,$fp_time" >> "$LOGFILE"

done


python3 graph_plot.py "$LOGFILE" "plot.png"

echo "Saved results to $LOGFILE"
