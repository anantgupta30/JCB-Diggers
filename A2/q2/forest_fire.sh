#!/usr/bin/env bash
# forest_fire.sh — Wrapper script for Forest Fire solver
# Usage: bash forest_fire.sh <graph_file> <seed_file> <output_file> <k> <n_random_instances> <hops>

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: bash forest_fire.sh <graph_file> <seed_file> <output_file> <k> <n_random_instances> <hops>"
    exit 1
fi

GRAPH_FILE="$1"
SEED_FILE="$2"
OUTPUT_FILE="$3"
K="$4"
N_RANDOM="$5"
HOPS="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/forest_fire.py" "$GRAPH_FILE" "$SEED_FILE" "$OUTPUT_FILE" "$K" "$N_RANDOM" "$HOPS"
