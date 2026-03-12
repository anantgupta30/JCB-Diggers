#!/usr/bin/env python3
"""
Forest Fire Spread Minimization — Reachability-Delta Greedy Solver
==================================================================
Selects exactly k edges to block in a probabilistic graph to minimize
expected fire spread σ(R) from seed set A0.

Strategy: Greedy selection using true marginal gain computed via
reachability-delta scoring over r Monte-Carlo live-edge samples.
"""

import sys
import random
from collections import defaultdict, deque


# ─────────────────────────────────────────────────────────────────────────────
# 1. I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(path):
    """Return adj dict {u: [(v, p), ...]}, set of all nodes, set of all edges."""
    adj = defaultdict(list)
    nodes = set()
    edges = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            adj[u].append((v, p))
            nodes.add(u)
            nodes.add(v)
            edges.add((u, v))
    return dict(adj), nodes, edges


def load_seeds(path):
    seeds = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            seeds.add(int(line))
    return seeds


def write_output(path, selected_edges):
    with open(path, 'w') as f:
        for u, v in selected_edges:
            f.write(f"{u} {v}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pruning — keep only nodes/edges reachable from A0 within h hops
# ─────────────────────────────────────────────────────────────────────────────

def prune_graph(adj, seeds, hops):
    """
    BFS from seeds on the full graph (ignoring probabilities) up to `hops` steps.
    Returns pruned adj dict and reachable node set.
    If hops is None (unlimited), do full BFS.
    """
    reachable = set(seeds)
    frontier = list(seeds)
    depth = 0
    while frontier:
        if hops is not None and depth >= hops:
            break
        next_frontier = []
        for u in frontier:
            for v, p in adj.get(u, []):
                if v not in reachable:
                    reachable.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        depth += 1

    # Build pruned adjacency — only edges where BOTH endpoints are reachable
    pruned_adj = {}
    for u in reachable:
        if u in adj:
            nbrs = [(v, p) for v, p in adj[u] if v in reachable]
            if nbrs:
                pruned_adj[u] = nbrs
    return pruned_adj, reachable


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sample live-edge subgraphs
# ─────────────────────────────────────────────────────────────────────────────

def sample_live_edge_graphs(adj, r, rng):
    """
    Generate r live-edge subgraphs.
    Each subgraph is stored as a dict {u: [v1, v2, ...]} (no probabilities).
    Returns list of subgraphs.
    """
    # Pre-collect all edges for fast sampling
    all_edges = []
    for u, nbrs in adj.items():
        for v, p in nbrs:
            all_edges.append((u, v, p))

    subgraphs = []
    for _ in range(r):
        g = defaultdict(list)
        for u, v, p in all_edges:
            if rng.random() < p:
                g[u].append(v)
        subgraphs.append(dict(g))
    return subgraphs


# ─────────────────────────────────────────────────────────────────────────────
# 4. BFS-based reachability
# ─────────────────────────────────────────────────────────────────────────────

def bfs_reachable(adj_graph, seeds, hops=None):
    """
    BFS on a live-edge graph (no probabilities). Returns the set of reachable nodes.
    hops=None means unlimited.
    """
    reached = set(seeds)
    frontier = list(seeds)
    depth = 0
    while frontier:
        if hops is not None and depth >= hops:
            break
        next_frontier = []
        for u in frontier:
            for v in adj_graph.get(u, []):
                if v not in reached:
                    reached.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        depth += 1
    return reached


def bfs_reachable_without_edge(adj_graph, seeds, block_u, block_v, hops=None):
    """
    BFS on a live-edge graph skipping one specific edge (block_u -> block_v).
    Returns the set of reachable nodes.
    """
    reached = set(seeds)
    frontier = list(seeds)
    depth = 0
    while frontier:
        if hops is not None and depth >= hops:
            break
        next_frontier = []
        for u in frontier:
            for v in adj_graph.get(u, []):
                if v not in reached:
                    if u == block_u and v == block_v:
                        continue
                    reached.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        depth += 1
    return reached


# ─────────────────────────────────────────────────────────────────────────────
# 5. Core: Compute baseline reachable sets and candidate edges
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline_reachable(subgraphs, seeds, hops):
    """Compute reachable set from seeds for each subgraph."""
    baselines = []
    for g in subgraphs:
        reached = bfs_reachable(g, seeds, hops)
        baselines.append(reached)
    return baselines


def get_candidate_edges(subgraphs, baselines, seeds):
    """
    Return dict: edge -> list of subgraph indices where that edge is live
    and in the reachable subgraph (i.e., source node is reachable).
    Edges originating from seeds are still candidates (blocking inbound to
    non-seeds or outbound from seeds both matter).
    """
    candidates = defaultdict(list)
    for i, g in enumerate(subgraphs):
        reached = baselines[i]
        for u in reached:
            for v in g.get(u, []):
                # Edge (u,v) is live and u is reachable
                candidates[(u, v)].append(i)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 6. Greedy selection with lazy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_marginal_gain(edge, subgraphs, baselines, seeds, hops, relevant_indices):
    """
    Compute the average marginal gain of blocking `edge` across relevant subgraphs.
    Only considers subgraphs where the edge is live and reachable.
    """
    u, v = edge
    total_gain = 0
    r = len(subgraphs)
    for i in relevant_indices:
        base_size = len(baselines[i])
        new_reached = bfs_reachable_without_edge(subgraphs[i], seeds, u, v, hops)
        gain = base_size - len(new_reached)
        total_gain += gain
    return total_gain / r


def greedy_select(adj, seeds, k, r, hops, rng):
    """
    Main greedy loop. Selects k edges to block.
    """
    print(f"  Sampling {r} live-edge subgraphs...", file=sys.stderr, flush=True)
    subgraphs = sample_live_edge_graphs(adj, r, rng)

    print(f"  Computing baseline reachability...", file=sys.stderr, flush=True)
    baselines = compute_baseline_reachable(subgraphs, seeds, hops)

    avg_baseline = sum(len(b) for b in baselines) / r
    print(f"  Avg baseline spread: {avg_baseline:.2f}", file=sys.stderr, flush=True)

    print(f"  Building candidate edge set...", file=sys.stderr, flush=True)
    candidates = get_candidate_edges(subgraphs, baselines, seeds)
    print(f"  Candidate edges: {len(candidates)}", file=sys.stderr, flush=True)

    selected = []

    for round_num in range(k):
        if not candidates:
            print(f"  Round {round_num+1}: No more candidate edges.", file=sys.stderr, flush=True)
            break

        print(f"  Round {round_num+1}/{k}: Evaluating {len(candidates)} candidates...",
              file=sys.stderr, flush=True)

        # --- Warm-start proxy for large candidate sets ---
        use_proxy = len(candidates) > 5 * k and k > 5
        if use_proxy:
            # Fast proxy: count how many subgraphs the edge appears in
            # weighted by the number of nodes reachable from v in each
            proxy_scores = {}
            for edge, indices in candidates.items():
                proxy_scores[edge] = len(indices)
            # Keep top 5k candidates
            shortlist_size = min(5 * k, len(candidates))
            top_edges = sorted(proxy_scores, key=proxy_scores.get, reverse=True)[:shortlist_size]
            eval_candidates = {e: candidates[e] for e in top_edges}
            print(f"    Proxy shortlisted {len(eval_candidates)} edges",
                  file=sys.stderr, flush=True)
        else:
            eval_candidates = candidates

        best_edge = None
        best_gain = -1

        for edge, indices in eval_candidates.items():
            if not indices:
                continue
            gain = compute_marginal_gain(edge, subgraphs, baselines, seeds, hops, indices)
            if gain > best_gain:
                best_gain = gain
                best_edge = edge

        if best_edge is None or best_gain <= 0:
            # If no edge provides gain, pick any remaining candidate
            if candidates:
                best_edge = next(iter(candidates))
                best_gain = 0.0
            else:
                break

        selected.append(best_edge)
        bu, bv = best_edge

        print(f"    Selected edge ({bu}, {bv}) with gain {best_gain:.4f}",
              file=sys.stderr, flush=True)

        # Remove selected edge from candidates
        if best_edge in candidates:
            del candidates[best_edge]

        # Update: remove the edge from all subgraphs and recompute reachability
        # only for affected subgraphs
        affected_indices = set()
        for i, g in enumerate(subgraphs):
            if bu in g and bv in g[bu]:
                g[bu].remove(bv)
                if not g[bu]:
                    del g[bu]
                affected_indices.add(i)

        # Recompute baselines only for affected subgraphs
        for i in affected_indices:
            baselines[i] = bfs_reachable(subgraphs[i], seeds, hops)

        # Rebuild candidate indices for affected subgraphs
        # Remove stale indices and add new ones
        edges_to_remove = []
        for edge in list(candidates.keys()):
            # Remove indices that were affected
            new_indices = [idx for idx in candidates[edge] if idx not in affected_indices]
            # Re-check affected subgraphs for this edge
            eu, ev = edge
            for i in affected_indices:
                if eu in baselines[i]:  # eu must be reachable
                    if eu in subgraphs[i] and ev in subgraphs[i].get(eu, []):
                        new_indices.append(i)
            candidates[edge] = new_indices
            if not new_indices:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            del candidates[edge]

    # If we selected fewer than k edges, pad with remaining candidates
    if len(selected) < k and candidates:
        remaining = sorted(candidates.keys(), key=lambda e: len(candidates[e]), reverse=True)
        for edge in remaining:
            if len(selected) >= k:
                break
            if edge not in selected:
                selected.append(edge)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 7:
        print("Usage: python forest_fire.py <graph_file> <seed_file> <output_file> <k> <r> <hops>",
              file=sys.stderr)
        sys.exit(1)

    graph_file = sys.argv[1]
    seed_file = sys.argv[2]
    output_file = sys.argv[3]
    k = int(sys.argv[4])
    r = int(sys.argv[5])
    hops_arg = int(sys.argv[6])
    hops = None if hops_arg < 0 else hops_arg

    print(f"Loading graph from {graph_file}...", file=sys.stderr, flush=True)
    adj, nodes, edges = load_graph(graph_file)
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}", file=sys.stderr, flush=True)

    seeds = load_seeds(seed_file)
    print(f"  Seeds: {len(seeds)} -> {sorted(seeds)[:10]}{'...' if len(seeds)>10 else ''}",
          file=sys.stderr, flush=True)

    # Aggressive pruning
    print(f"Pruning graph (hops={hops_arg})...", file=sys.stderr, flush=True)
    pruned_adj, reachable = prune_graph(adj, seeds, hops)
    pruned_edges = sum(len(nbrs) for nbrs in pruned_adj.values())
    print(f"  Pruned: {len(reachable)} nodes, {pruned_edges} edges",
          file=sys.stderr, flush=True)

    rng = random.Random(42)

    print(f"Running greedy selection (k={k}, r={r}, hops={hops_arg})...",
          file=sys.stderr, flush=True)
    selected = greedy_select(pruned_adj, seeds, k, r, hops, rng)

    # Ensure exactly k edges in output
    # If we have fewer than k, pad with dummy edges from the edge set
    if len(selected) < k:
        # Pick remaining edges from the pruned graph that aren't already selected
        selected_set = set(selected)
        for u, nbrs in pruned_adj.items():
            if len(selected) >= k:
                break
            for v, p in nbrs:
                if (u, v) not in selected_set:
                    selected.append((u, v))
                    selected_set.add((u, v))
                    if len(selected) >= k:
                        break
    # If still not enough (unlikely), pad from original graph
    if len(selected) < k:
        selected_set = set(selected)
        for u, nbrs in adj.items():
            if len(selected) >= k:
                break
            for v, p in nbrs:
                if (u, v) not in selected_set:
                    selected.append((u, v))
                    selected_set.add((u, v))
                    if len(selected) >= k:
                        break

    selected = selected[:k]
    write_output(output_file, selected)
    print(f"Output written to {output_file} ({len(selected)} edges)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
