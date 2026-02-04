import sys
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from utils import parse_dataset

def build_nx_graph(g_dict):
    """
    Converts dictionary graph representation to NetworkX graph.
    Nodes must have 'label' attribute.
    Edges must have 'label' attribute.
    """
    G = nx.Graph()
    for nid, label in g_dict['nodes'].items():
        G.add_node(nid, label=str(label))
    
    for u, v, label in g_dict['edges']:
        G.add_edge(u, v, label=str(label))
    return G

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 convert.py <graphs_path> <features_path> <output_npy_path>")
        sys.exit(1)

    graphs_path = sys.argv[1]
    features_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load Graphs
    db_graphs = parse_dataset(graphs_path)
    feature_graphs = parse_dataset(features_path)
    
    # Normalize labels to integers for matching
    def normalize_labels(g):
        new_nodes = {}
        for nid, lbl in g['nodes'].items():
            try: new_nodes[nid] = int(lbl)
            except: new_nodes[nid] = lbl
        new_edges = []
        for u, v, lbl in g['edges']:
            try: new_edges.append((u, v, int(lbl)))
            except: new_edges.append((u, v, lbl))
        g['nodes'] = new_nodes
        g['edges'] = new_edges
        return g

    print("Normalizing labels to integers for matching...")
    db_graphs = [normalize_labels(g) for g in db_graphs]
    feature_graphs = [normalize_labels(g) for g in feature_graphs]

    print(f"Loaded {len(db_graphs)} database graphs and {len(feature_graphs)} features.")

    # Convert to NetworkX
    nx_db = [build_nx_graph(g) for g in db_graphs]
    nx_feats = [build_nx_graph(g) for g in feature_graphs]

    # Node and Edge Matchers for Isomorphism
    nm = isomorphism.categorical_node_match("label", None)
    em = isomorphism.categorical_edge_match("label", None)

    # Feature Matrix: Graphs x Features
    matrix = np.zeros((len(nx_db), len(nx_feats)), dtype=int)

    for i, G in enumerate(nx_db):
        for j, F in enumerate(nx_feats):
            # Check if Feature F is a subgraph of Graph G
            GM = isomorphism.GraphMatcher(G, F, node_match=nm, edge_match=em)
            if GM.subgraph_is_isomorphic():
                matrix[i, j] = 1
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(nx_db)} graphs...")

    # Save to file
    np.save(output_path, matrix)
    print(f"Feature matrix saved to {output_path}")

if __name__ == "__main__":
    main()
