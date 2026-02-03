import sys
import networkx as nx
import multiprocessing
from multiprocessing import Pool, cpu_count
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

# --- GLOBAL WORKER MEMORY ---
# This ensures patterns are sent to workers only once, not 40,000 times.
GLOBAL_PATTERNS = []
GLOBAL_MAX_LEN = 8

def init_worker(patterns, max_len):
    """Initialize worker with read-only global data."""
    global GLOBAL_PATTERNS, GLOBAL_MAX_LEN
    GLOBAL_PATTERNS = patterns
    GLOBAL_MAX_LEN = max_len

# --- HELPER FUNCTIONS ---
def get_canonical_path(path_nodes, graph):
    forward = []
    for i, node in enumerate(path_nodes):
        forward.append(graph.nodes[node]['label'])
        if i < len(path_nodes) - 1:
            u, v = node, path_nodes[i+1]
            edge_label = graph.edges[u, v].get('label', '0')
            forward.append(f"-{edge_label}-")
    forward_str = "".join(forward)
    
    # Optimization: Only build backward string if forward isn't palindromic-ish
    # But for safety and canonical correctness, we keep full check
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

def process_graph_signatures(args):
    # Unpack only the graph index and graph object
    # Patterns are accessed from global memory
    G_idx, G = args
    
    local_path_sigs = set()
    local_ring_sigs = set()
    
    # 1. PATHS
    for start_node in G.nodes():
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            path_len = len(path) - 1
            if path_len >= 0:
                local_path_sigs.add(get_canonical_path(path, G))
            if path_len >= GLOBAL_MAX_LEN: continue
            for neighbor in G.neighbors(curr):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    
    # 2. RINGS
    try:
        # to_directed() is required for simple_cycles to work on undirected graphs
        for ring_nodes in nx.simple_cycles(G.to_directed()):
            if 3 <= len(ring_nodes) <= 12:
                local_ring_sigs.add(get_ring_hash(G, ring_nodes))
    except: pass

    # 3. GENERATE BIT VECTOR
    bit_vector = []
    for ptype, psig in GLOBAL_PATTERNS:
        if ptype == 'PATH':
            bit_vector.append('1' if psig in local_path_sigs else '0')
        elif ptype == 'RING':
            bit_vector.append('1' if psig in local_ring_sigs else '0')
        else:
            bit_vector.append('0')
            
    return " ".join(bit_vector)

def parse_graph_file(filename):
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
            if parts[0] == 't' or parts[0] == '#':
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

def main():
    if len(sys.argv) < 4:
        print("Usage: python vectorizer.py <graphs> <patterns_file> <output_vecs>")
        sys.exit(1)
    db_file = sys.argv[1]
    pattern_file = sys.argv[2]
    out_file = sys.argv[3]
    
    sorted_sigs = []
    with open(pattern_file, 'r') as f:
        for line in f:
            if line.startswith("# SIG:"):
                content = line.strip().split(" ", 2)[2]
                if " (Supp:" in content: content = content.split(" (Supp:", 1)[0]
                parts = content.split("::", 1)
                sorted_sigs.append((parts[0], parts[1]))
                
    db_graphs = parse_graph_file(db_file)
    n_cores = cpu_count()
    
    print(f"Vectorizing {len(db_graphs)} graphs on {n_cores} cores...")

    # We only pass (index, graph) to the worker
    tasks = [(i, G) for i, G in enumerate(db_graphs)]
    
    # Use initializer to share patterns efficiently
    with Pool(n_cores, initializer=init_worker, initargs=(sorted_sigs, 8)) as p:
        # chunksize=50 helps balance load without excessive queuing
        results = p.imap(process_graph_signatures, tasks, chunksize=50)
        
        with open(out_file, 'w') as f:
            for i, vec_str in enumerate(results):
                f.write(vec_str + "\n")
                if i % 1000 == 0:
                    sys.stderr.write(f"\rProgress: {i}/{len(db_graphs)}")
    print("\nDone.")

if __name__ == "__main__":
    main()