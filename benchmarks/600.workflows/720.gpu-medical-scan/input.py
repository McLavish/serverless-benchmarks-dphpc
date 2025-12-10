size_generators = {
    "test": (10, 128, 128, 3),  # 10 slices, 128x128 resolution, 3 organs
    "small": (25, 256, 256, 5),
    "large": (50, 512, 512, 8),
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
    n_slices, height, width, n_organs = size_generators[size]
    return {
        "n_slices": n_slices,
        "height": height,
        "width": width,
        "n_organs": n_organs,
    }
