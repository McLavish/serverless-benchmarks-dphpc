import os
import json
from typing import Tuple


def buckets_count() -> Tuple[int, int]:
    """
    One input bucket, one output bucket.
    Same convention as other SeBS benchmarks.
    """
    return (1, 1)


def generate_input(
    data_dir,
    size,
    benchmarks_bucket,
    input_paths,
    output_paths,
    upload_func,
    nosql_func,
):
    """
    Generate a JSON config controlling the PageRank benchmark.

    We keep the graph synthetic to avoid large input files; all heavy
    work happens inside the function.

    Size presets:
      - test:  1k nodes,  4 avg degree, 10 iterations
      - small: 10k nodes, 8 avg degree, 10 iterations
      - large: 50k nodes, 8 avg degree, 20 iterations
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "benchmarks-data", "700.graphs", "710.graph-pagerank")

    if size == "test":
        num_nodes = 1_000
        avg_degree = 4
        iterations = 10
    elif size == "small":
        num_nodes = 10_000
        avg_degree = 8
        iterations = 10
    else:  # "large"
        num_nodes = 50_000
        avg_degree = 8
        iterations = 20

    config = {
        "num_nodes": num_nodes,
        "avg_degree": avg_degree,
        "iterations": iterations,
        "damping": 0.85,
        "seed": 42,
    }

    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, f"graph_pagerank_{size}.json")
    with open(local_path, "w") as f:
        json.dump(config, f)

    # Upload JSON to input bucket 0 under the given prefix
    key = f"{input_paths[0].rstrip('/')}/graph_pagerank_{size}.json"
    upload_func(0, key, local_path)

    # Return the config directly so local/HTTP invocations can run without
    # fetching it back from storage.
    return {"config": config}
