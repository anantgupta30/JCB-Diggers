import sys
import argparse
import networkx as nx
from collections import Counter
import multiprocessing
from functools import partial
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

# --- HELPER FUNCTIONS ---
def load_graphs_robust(filename):
    graphs = []
    current_graph = None
    try:
        f = open(filename, 'r', encoding='utf-8-sig')
    except:
        f = open(filename, 'r', encoding='cp1252')
    with f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if parts[0] == '#' or parts[0] == 't':
                if current_graph: graphs.append(current_graph)
                current_graph = nx.Graph()
            elif parts[0] == 'v':
                if current_graph is None: current_graph = nx.Graph()
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                if current_graph:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    if current_graph: graphs.append(current_graph)
    return graphs

def get_canonical_path(path_nodes, graph):
    forward = []
    for i, node in enumerate(path_nodes):
        forward.append(graph.nodes[node]['label'])
        if i < len(path_nodes) - 1:
            u, v = node, path_nodes[i+1]
            edge_label = graph.edges[u, v].get('label', '0')
            forward.append(f"-{edge_label}-")
    forward_str = "".join(forward)
    backward_path_nodes = path_nodes[::-1]
    backward = []
    for i, node in enumerate(backward_path_nodes):
        backward.append(graph.nodes[node]['label'])
        if i < len(backward_path_nodes) - 1:
            u, v = node, backward_path_nodes[i+1]
            edge_label = graph.edges[u, v].get('label', '0')
            backward.append(f"-{edge_label}-")
    backward_str = "".join(backward)
    return min(forward_str, backward_str)

def build_hollow_ring(graph, ring_nodes):
    cycle_subgraph = nx.Graph()
    for n in ring_nodes:
        cycle_subgraph.add_node(n, label=graph.nodes[n]['label'])
    for i in range(len(ring_nodes)):
        u = ring_nodes[i]
        v = ring_nodes[(i + 1) % len(ring_nodes)]
        if graph.has_edge(u, v):
            edge_label = graph.edges[u, v].get('label', '0')
            cycle_subgraph.add_edge(u, v, label=edge_label)
    return cycle_subgraph

def get_ring_hash(graph, ring_nodes):
    sub = build_hollow_ring(graph, ring_nodes)
    return weisfeiler_lehman_graph_hash(sub, node_attr='label', iterations=1)

def mine_chunk(graphs_chunk, max_length=6):
    path_counts = {i: Counter() for i in range(max_length + 1)}
    ring_counts = Counter()
    ring_protos = {} 
    for G in graphs_chunk:
        found_paths = {i: set() for i in range(max_length + 1)}
        for start_node in G.nodes():
            stack = [(start_node, [start_node])]
            while stack:
                curr, path = stack.pop()
                path_len = len(path) - 1
                if path_len >= 0:
                    found_paths[path_len].add(get_canonical_path(path, G))
                if path_len >= max_length: continue
                for neighbor in G.neighbors(curr):
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        for length in range(max_length + 1):
            for sig in found_paths[length]:
                path_counts[length][sig] += 1
        found_rings = set()
        try:
            for ring_nodes in nx.simple_cycles(G.to_directed()):
                if 3 <= len(ring_nodes) <= 12:
                    h = get_ring_hash(G, ring_nodes)
                    if h not in found_rings:
                        found_rings.add(h)
                        if h not in ring_protos:
                            ring_protos[h] = build_hollow_ring(G, ring_nodes)
            for h in found_rings:
                ring_counts[h] += 1
        except: pass
    return path_counts, ring_counts, ring_protos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_db')
    parser.add_argument('output_features')
    parser.add_argument('--min_r', type=float, default=0.01) 
    parser.add_argument('--max_r', type=float, default=0.60)
    parser.add_argument('--len', type=int, default=6)
    parser.add_argument('--k', type=int, default=50, help='Number of top features to select')
    args = parser.parse_args()
    
    db_graphs = load_graphs_robust(args.input_db)
    total_graphs = len(db_graphs)
    n_cores = multiprocessing.cpu_count()
    
    # Chunking Optimization:
    # Ensure at least 4 chunks per core to allow load balancing
    # But ensure chunk isn't too small (<10 graphs) to avoid overhead
    chunk_size = max(10, total_graphs // (n_cores * 4))
    
    chunks = [db_graphs[i:i + chunk_size] for i in range(0, total_graphs, chunk_size)]
    
    global_paths = {i: Counter() for i in range(args.len + 1)}
    global_rings = Counter()
    global_ring_protos = {}
    
    worker = partial(mine_chunk, max_length=args.len)
    print(f"Mining on {n_cores} cores with {len(chunks)} chunks...")
    
    with multiprocessing.Pool(n_cores) as pool:
        for i, (p_counts, r_counts, r_protos) in enumerate(pool.imap_unordered(worker, chunks)):
            for length in p_counts:
                global_paths[length].update(p_counts[length])
            global_rings.update(r_counts)
            global_ring_protos.update(r_protos)
            if i % 10 == 0:
                sys.stderr.write(f"\rProgress: {i}/{len(chunks)}")

    all_candidates = []
    min_count = int(total_graphs * args.min_r)
    max_count = int(total_graphs * args.max_r)

    def path_tuple_to_graph(sig_str):
        G = nx.Graph()
        tokens = sig_str.replace('-', ' ').split()
        if len(tokens) == 1:
            G.add_node(0, label=tokens[0])
            return G
        node_idx = 0
        G.add_node(node_idx, label=tokens[0])
        curr_idx = 1
        while curr_idx < len(tokens):
            edge_label = tokens[curr_idx]
            next_node = tokens[curr_idx+1]
            node_idx += 1
            G.add_node(node_idx, label=next_node)
            G.add_edge(node_idx-1, node_idx, label=edge_label)
            curr_idx += 2
        return G

    for h, count in global_rings.items():
        if min_count <= count <= max_count:
            all_candidates.append((count, 'RING', h, global_ring_protos[h]))
    for length in range(args.len + 1):
        for sig, count in global_paths[length].items():
            if min_count <= count <= max_count:
                all_candidates.append((count, 'PATH', sig, None))

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = all_candidates[:args.k]

    with open(args.output_features, 'w') as f:
        for i, (count, ptype, sig, proto) in enumerate(top_k):
            G = proto
            if ptype == 'PATH': G = path_tuple_to_graph(sig)
            f.write(f"# SIG: {ptype}::{sig} (Supp: {count})\n")
            f.write(f"t # {i}\n")
            mapping = {n: j for j, n in enumerate(sorted(G.nodes()))}
            for n in sorted(G.nodes()):
                f.write(f"v {mapping[n]} {G.nodes[n]['label']}\n")
            for u, v in sorted(G.edges()):
                label = G.edges[u,v].get('label', '0')
                f.write(f"e {mapping[u]} {mapping[v]} {label}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()