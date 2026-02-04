import sys

def get_graph_hash(g):
    """
    Returns a canonical string representation of a graph for duplicate detection.
    """
    nodes_str = ",".join([f"{lbl}" for nid, lbl in sorted(g['nodes'].items())])
    # For edges, we sort them canonically
    edges = []
    for u, v, lbl in g['edges']:
        u_lbl = g['nodes'][u]
        v_lbl = g['nodes'][v]
        # Store edge as (sorted_node_labels, edge_label)
        edges.append(tuple(sorted([u_lbl, v_lbl])) + (lbl,))
    edges_str = ",".join([f"{e[0]}-{e[1]}:{e[2]}" for e in sorted(edges)])
    return f"N[{nodes_str}]|E[{edges_str}]"

def parse_dataset(filepath):
    graphs = []
    current_graph = None
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        graph_ct = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('#') or line.startswith('t'):
                if current_graph is not None:
                    if 'seen_edges' in current_graph: del current_graph['seen_edges']
                    graphs.append(current_graph)
                current_graph = {'id': graph_ct, 'nodes': {}, 'edges': []}
                graph_ct += 1
            elif line.startswith('v'):
                parts = line.split()
                # v <id> <label>
                node_id = int(parts[1])
                label = parts[2]
                current_graph['nodes'][node_id] = label
            elif line.startswith('e'):
                parts = line.split()
                # e <u_id> <v_id> <label>
                u, v, label = int(parts[1]), int(parts[2]), parts[3]
                # In undirected graphs, only store each edge once (u < v)
                u_id, v_id = min(u, v), max(u, v)
                edge_sig = (u_id, v_id)
                if 'seen_edges' not in current_graph:
                    current_graph['seen_edges'] = set()
                if edge_sig not in current_graph['seen_edges']:
                    current_graph['edges'].append((u_id, v_id, label))
                    current_graph['seen_edges'].add(edge_sig)
                    
        if current_graph is not None:
            if 'seen_edges' in current_graph: del current_graph['seen_edges']
            graphs.append(current_graph)
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
        
    return graphs

def write_gspan(graphs, output_path):
    node_lbl_map = {}
    edge_lbl_map = {}
    next_node_lbl = 1
    next_edge_lbl = 1

    def get_lbl(lbl, lmap, next_id):
        try:
            return int(lbl), next_id
        except:
            if lbl not in lmap:
                lmap[lbl] = next_id
                return next_id, next_id + 1
            return lmap[lbl], next_id

    with open(output_path, 'w') as f:
        for g in graphs:
            f.write(f"t # {g['id']}\n")
            
            # Sort for stability
            sorted_nodes = sorted(g['nodes'].items())
            for nid, label in sorted_nodes:
                ilbl, next_node_lbl = get_lbl(label, node_lbl_map, next_node_lbl)
                f.write(f"v {nid} {ilbl}\n")
                
            for u, v, label in g['edges']:
                ilbl, next_edge_lbl = get_lbl(label, edge_lbl_map, next_edge_lbl)
                f.write(f"e {u} {v} {ilbl}\n")
    
    # Save maps to file for identification to use when writing back
    import json
    with open(output_path + '.map', 'w') as f_map:
        json.dump({'nodes': node_lbl_map, 'edges': edge_lbl_map}, f_map)
    
    return None, None

def parse_gspan_fp(filepath, map_path=None):
    node_rev_map = {}
    edge_rev_map = {}
    if map_path:
        import json
        with open(map_path, 'r') as f:
            data = json.load(f)
            node_rev_map = {int(v): k for k, v in data['nodes'].items()}
            edge_rev_map = {int(v): k for k, v in data['edges'].items()}

    graphs = []
    current_graph = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    last_support = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('#'):
             try:
                 parts = line.split()
                 if len(parts) >= 2:
                     last_support = int(parts[1])
             except:
                 pass
             continue
        if line.startswith('t'):
            if current_graph is not None:
                graphs.append(current_graph)
            parts = line.split()
            if parts[1] == '#':
                 gid = int(parts[2])
                 if '*' in parts:
                     idx = parts.index('*')
                     if idx + 1 < len(parts):
                         last_support = int(parts[idx+1])
            else:
                 gid = int(parts[1])
            
            current_graph = {'id': gid, 'nodes': {}, 'edges': [], 'support': last_support}
        elif line.startswith('v'):
            parts = line.split()
            if len(parts) < 3: continue
            nid = int(parts[1])
            ilbl = int(parts[2])
            # Restore original label if mapped
            label = node_rev_map.get(ilbl, str(ilbl))
            current_graph['nodes'][nid] = label 
        elif line.startswith('e'):
            parts = line.split()
            if len(parts) < 4: continue
            u = int(parts[1])
            v = int(parts[2])
            ilbl = int(parts[3])
            # Restore original label if mapped
            label = edge_rev_map.get(ilbl, str(ilbl))
            current_graph['edges'].append((u, v, label))
            
    if current_graph is not None:
        graphs.append(current_graph)
        
    return graphs
