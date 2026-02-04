#!/bin/bash
# generate_candidates.sh <path_database_graph_features> <path_query_graph_features> <path_out_file>

if [ "$#" -ne 3 ]; then
    echo "Usage: ./generate_candidates.sh <path_database_graph_features> <path_query_graph_features> <path_out_file>"
    exit 1
fi

DB_FEATURES=$1
QUERY_FEATURES=$2
OUT_FILE=$3

echo "Generating candidate sets..."
python3 generate_candidates.py "$DB_FEATURES" "$QUERY_FEATURES" "$OUT_FILE"
