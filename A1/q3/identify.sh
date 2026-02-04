#!/bin/bash
# identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

if [ "$#" -ne 2 ]; then
    echo "Usage: ./identify.sh <path_graph_dataset> <path_discriminative_subgraphs>"
    exit 1
fi

INPUT_DATASET=$1
OUTPUT_FILE=$2

# Create temporary files
GSPAN_INPUT="temp_dataset.gspan"
GSPAN_OUTPUT="temp_features.gspan"

echo "Running ultra-optimized variance-based feature identification..."

# 1. Duplicate Removal & Pre-filtering
python3 -c "
import sys
import os
from utils import parse_dataset, write_gspan, get_graph_hash

graphs = parse_dataset('$INPUT_DATASET')
print(f'Original database size: {len(graphs)}')

unique_graphs = []
seen_hashes = set()
for g in graphs:
    h = get_graph_hash(g)
    if h not in seen_hashes:
        unique_graphs.append(g)
        seen_hashes.add(h)

print(f'Unique graphs for mining: {len(unique_graphs)}')
write_gspan(unique_graphs, '$GSPAN_INPUT')
"

# 2. Run Gaston at extremely low support
TOTAL_GRAPHS=$(grep -c "^t" "$GSPAN_INPUT")
if [ "$TOTAL_GRAPHS" -gt 10000 ]; then
    # 1.5% support for large datasets
    SUPPORT_ABS=$(( TOTAL_GRAPHS * 15 / 1000 ))
else
    # 4% support for smaller datasets
    SUPPORT_ABS=$(( TOTAL_GRAPHS * 4 / 100 ))
fi
if [ "$SUPPORT_ABS" -lt 1 ]; then SUPPORT_ABS=1; fi

echo "Mining with Gaston (Support: $SUPPORT_ABS, Total unique: $TOTAL_GRAPHS)..."

if [ ! -f "./gaston" ]; then
    echo "Error: ./gaston binary not found."
    exit 1
fi
chmod +x ./gaston

python3 -c "
import subprocess
try:
    subprocess.run(['./gaston', '$SUPPORT_ABS', '$GSPAN_INPUT', '$GSPAN_OUTPUT'], timeout=120)
except subprocess.TimeoutExpired:
    print('Gaston timed out. Using partial results.')
"

# 3. Greedy Bit-Vector Orthogonalization
# We select 50 features that are high variance AND low correlation
python3 -c "
import sys
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from utils import parse_dataset, parse_gspan_fp

def build_nx_graph(g_dict):
    G = nx.Graph()
    for nid, label in g_dict['nodes'].items():
        G.add_node(nid, label=str(label))
    for u, v, label in g_dict['edges']:
        G.add_edge(u, v, label=str(label))
    return G

# Load Features and Database Sample
features = parse_gspan_fp('$GSPAN_OUTPUT', map_path='$GSPAN_INPUT.map')
db_graphs = parse_dataset('$GSPAN_INPUT')
total_db = len(db_graphs)
target_freq = total_db / 2.0

# Filter features by size and local support
features = [f for f in features if 2 <= len(f['edges']) <= 9]
if not features: 
    features = parse_gspan_fp('$GSPAN_OUTPUT')[:200]

# Pre-rank by variance (Entropy)
features.sort(key=lambda x: abs(x.get('support', 0) - target_freq))
candidates = features[:300] # Analyze top 300 for diversity

# Sample DB for bit-vector calculation (performance)
sample_size = min(total_db, 500)
db_sample = db_graphs[:sample_size]
nx_db = [build_nx_graph(g) for g in db_sample]
nm = isomorphism.categorical_node_match('label', None)
em = isomorphism.categorical_edge_match('label', None)

print(f'Calculating bit-vectors for {len(candidates)} features on {sample_size} graphs...')
bitvectors = []
valid_candidates = []

for f in candidates:
    F = build_nx_graph(f)
    vec = np.zeros(sample_size, dtype=int)
    for i, G in enumerate(nx_db):
        GM = isomorphism.GraphMatcher(G, F, node_match=nm, edge_match=em)
        if GM.subgraph_is_isomorphic():
            vec[i] = 1
    # Only keep if it varies at all in the sample
    if np.any(vec) and not np.all(vec):
        bitvectors.append(vec)
        valid_candidates.append(f)

# Greedy Selection: Pick features that maximize collective entropy
selected_indices = []
selected_vecs = []

if valid_candidates:
    # 1st: Max Variance
    vars = [np.var(v) for v in bitvectors]
    best_idx = np.argmax(vars)
    selected_indices.append(best_idx)
    selected_vecs.append(bitvectors[best_idx])
    
    for _ in range(49):
        if len(selected_indices) >= len(valid_candidates): break
        
        best_score = -1
        best_f_idx = -1
        
        for i, v in enumerate(bitvectors):
            if i in selected_indices: continue
            
            # Score = Variance * (1 - MaxCorrelation)
            # Correlation approximated by dot product overlap
            v_var = np.var(v)
            max_overlap = 0
            for sv in selected_vecs:
                # Jaccard-like similarity
                overlap = np.sum(v & sv) / max(1, np.sum(v | sv))
                if overlap > max_overlap: max_overlap = overlap
            
            score = v_var * (1.0 - max_overlap)
            if score > best_score:
                best_score = score
                best_f_idx = i
        
        if best_f_idx != -1:
            selected_indices.append(best_f_idx)
            selected_vecs.append(bitvectors[best_f_idx])

selected = [valid_candidates[i] for i in selected_indices]
if len(selected) < 50:
    # Fill with remaining from original features
    for f in features:
        if f not in selected:
            selected.append(f)
            if len(selected) >= 50: break

print(f'Selected {len(selected)} variance-orthogonalized features.')

with open('$OUTPUT_FILE', 'w') as f:
    for g in selected:
        f.write(f'# {g[\"id\"]}\\n')
        nodes = sorted(g['nodes'].items())
        for nid, lbl in nodes:
            f.write(f'v {nid} {lbl}\\n')
        for u, v, lbl in g['edges']:
            f.write(f'e {u} {v} {lbl}\\n')
"

echo "Selected $(grep -c "^#" $OUTPUT_FILE) discriminative subgraphs."
rm "$GSPAN_INPUT" "$GSPAN_OUTPUT" "$GSPAN_INPUT.map"
