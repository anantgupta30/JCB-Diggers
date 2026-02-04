#!/bin/bash
# convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

if [ "$#" -ne 3 ]; then
    echo "Usage: ./convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
    exit 1
fi

GRAPHS=$1
SUBGRAPHS=$2
FEATURES_OUT=$3

echo "Converting graphs to feature vectors..."
mkdir -p "$(dirname "$FEATURES_OUT")" 2>/dev/null
python3 convert.py "$GRAPHS" "$SUBGRAPHS" "$FEATURES_OUT"
