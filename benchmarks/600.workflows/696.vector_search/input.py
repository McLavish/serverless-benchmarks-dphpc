size_generators = {
    # n_queries, embedding_dim, top_k
    "test": (5, 64, 3),
    "small": (20, 128, 5),
    "large": (50, 256, 10),
}


def buckets_count():
    # No object storage buckets required for this workflow.
    return (0, 0)


def generate_input(
    data_dir,
    size,
    benchmarks_bucket,
    input_buckets,
    output_buckets,
    upload_func,
    nosql_func,
):
    n_queries, embedding_dim, top_k = size_generators[size]
    return {
        "n_queries": n_queries,
        "embedding_dim": embedding_dim,
        "top_k": top_k,
    }
