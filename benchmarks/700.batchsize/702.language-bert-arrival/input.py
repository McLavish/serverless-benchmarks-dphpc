import copy
import os


SIZE_PROFILES = {
    "test": {
        "arrival_rate_rps": 8,
        "simulation_duration_s": 3,
        "latency_budget_ms": 100,
        "max_batch_size": 8,
        "warmup_runs": 1,
        "report_samples": 2,
    },
    "small": {
        "arrival_rate_rps": 16,
        "simulation_duration_s": 6,
        "latency_budget_ms": 200,
        "max_batch_size": 16,
        "warmup_runs": 2,
        "report_samples": 4,
    },
    "large": {
        "arrival_rate_rps": 24,
        "simulation_duration_s": 10,
        "latency_budget_ms": 300,
        "max_batch_size": 32,
        "warmup_runs": 3,
        "report_samples": 6,
    },
}


def buckets_count():
    # model bucket and text bucket
    return (2, 0)


def upload_files(data_root, data_dir, upload_func):
    for root, _, files in os.walk(data_dir):
        prefix = os.path.relpath(root, data_root)
        for file in files:
            filepath = os.path.join(root, file)
            relative_key = os.path.join(prefix, file)
            upload_func(0, relative_key, filepath)


def generate_input(
    data_dir, size, benchmarks_bucket, input_paths, output_paths, upload_func, nosql_func
):
    model_archive = "bert-tiny-onnx.tar.gz"
    upload_func(0, model_archive, os.path.join(data_dir, "model", model_archive))

    text_filename = "sentences.jsonl"
    upload_func(1, text_filename, os.path.join(data_dir, "text", text_filename))

    size_profile = copy.deepcopy(SIZE_PROFILES.get(size, SIZE_PROFILES["test"]))

    input_config = {"object": {}, "bucket": {}, "experiment": size_profile}
    input_config["object"]["model"] = model_archive
    input_config["object"]["input"] = text_filename
    input_config["bucket"]["bucket"] = benchmarks_bucket
    input_config["bucket"]["model"] = input_paths[0]
    input_config["bucket"]["text"] = input_paths[1]
    return input_config
