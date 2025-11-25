import json
import time
from typing import Any, Dict

import numpy as np


def generate_random_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    """
    Generate a random directed graph with fixed out-degree per node.

    Returns:
      - sources: int64 array of shape (E,)
      - targets: int64 array of shape (E,)
      - outdeg:  int64 array of shape (N,)
    """
    rng = np.random.default_rng(seed)

    # total edges
    edges_per_node = max(1, int(avg_degree))
    num_edges = num_nodes * edges_per_node

    # each node has exactly edges_per_node outgoing edges
    sources = np.repeat(np.arange(num_nodes, dtype=np.int64), edges_per_node)
    targets = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)

    outdeg = np.full(num_nodes, edges_per_node, dtype=np.int64)

    return sources, targets, outdeg


def pagerank_power_iteration(
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    outdeg: np.ndarray,
    iterations: int,
    damping: float,
):
    """
    Simple power-iteration PageRank using edge lists.

    For each iteration:
      new_rank = (1 - d) / N
      new_rank[v] += d * sum_{u->v} rank[u] / outdeg[u]
    """
    n = num_nodes
    d = damping

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    new_rank = np.empty_like(rank)

    for _ in range(iterations):
        # base teleportation term
        new_rank.fill((1.0 - d) / n)

        # contribution from neighbors
        contrib = d * (rank / outdeg)
        # accumulate contributions along edges
        np.add.at(new_rank, targets, contrib[sources])

        # swap
        rank, new_rank = new_rank, rank

    return rank


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    num_nodes = int(config["num_nodes"])
    avg_degree = int(config["avg_degree"])
    iterations = int(config["iterations"])
    damping = float(config.get("damping", 0.85))
    seed = int(config.get("seed", 42))

    # Generate graph
    t0 = time.perf_counter()
    sources, targets, outdeg = generate_random_graph(num_nodes, avg_degree, seed)

    # Run PageRank iterations
    t1 = time.perf_counter()
    ranks = pagerank_power_iteration(
        num_nodes, sources, targets, outdeg, iterations, damping
    )
    t2 = time.perf_counter()

    # some sanity to avoid "unused" elimination (even though Python won't)
    checksum = float(ranks.sum())

    graph_gen_ms = (t1 - t0) * 1e3
    pagerank_ms = (t2 - t1) * 1e3
    total_ms = (t2 - t0) * 1e3

    num_edges = int(len(sources))
    edges_per_iter = num_edges
    total_edges_processed = edges_per_iter * iterations
    edges_per_sec = (
        total_edges_processed / (pagerank_ms / 1e3) if pagerank_ms > 0 else 0.0
    )

    return {
        "num_nodes": num_nodes,
        "avg_degree": avg_degree,
        "iterations": iterations,
        "damping": damping,
        "num_edges": num_edges,
        "graph_generation_ms": graph_gen_ms,
        "pagerank_ms": pagerank_ms,
        "total_ms": total_ms,
        "edges_per_sec": edges_per_sec,
        "checksum": checksum,
    }


def handler(event, context=None):
    """
    SeBS entry point.

    Event is the JSON config uploaded by input.py, e.g.:

      {
        "num_nodes": 10000,
        "avg_degree": 8,
        "iterations": 10,
        "damping": 0.85,
        "seed": 42
      }
    """
    if isinstance(event, str):
        event = json.loads(event)

    # Some platforms may wrap config under 'config'
    if isinstance(event, dict) and "config" in event and isinstance(event["config"], dict):
        config = event["config"]
    else:
        config = event

    result = run_benchmark(config)
    return result
