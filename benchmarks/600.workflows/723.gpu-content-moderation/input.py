size_generators = {
    "test": (30, 100),   # 30 posts, avg 100 tokens per post
    "small": (100, 200),
    "large": (250, 300),
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
    n_posts, avg_tokens = size_generators[size]
    return {
        "n_posts": n_posts,
        "avg_tokens": avg_tokens,
    }
