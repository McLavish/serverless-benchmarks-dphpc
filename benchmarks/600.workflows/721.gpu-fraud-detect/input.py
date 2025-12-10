size_generators = {
    "test": (50, 20),   # 50 transactions, 20 features per transaction
    "small": (200, 30),
    "large": (500, 50),
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
    n_transactions, n_features = size_generators[size]
    return {
        "n_transactions": n_transactions,
        "n_features": n_features,
    }
