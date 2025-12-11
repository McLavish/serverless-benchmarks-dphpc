#!/usr/bin/env python3
import torch
import datetime


def initialize_torch(size, dtype=torch.float16, device="cuda"):
    a = torch.tensor(1.5, dtype=dtype, device=device)
    x = torch.randn(size, dtype=dtype, device=device)
    y = torch.randn(size, dtype=dtype, device=device)
    return a, x, y


def handler(event):

    size = event.get("size", 1000000)
    # reps = event.get("reps", 1000)
    # total elements processed across all launches
    total_elems = 1e8
    reps = total_elems // size

    if "seed" in event:
        import random

        random.seed(event["seed"])
        seed = event.get("seed")
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    sync = torch.cuda.synchronize

    vector_generating_begin = datetime.datetime.now()
    sync()
    a, x, y = initialize_torch(size, dtype=torch.float16, device="cuda")
    sync()
    vector_generating_end = datetime.datetime.now()

    # Warm up
    for _ in range(10):
        y = a * x + y
    sync()

    axpy_begin = datetime.datetime.now()
    sync()
    for _ in range(reps):
        y = a * x + y
    sync()
    axpy_end = datetime.datetime.now()

    vector_generating_time = (vector_generating_end - vector_generating_begin) / datetime.timedelta(
        microseconds=1
    )
    axpy_time = (axpy_end - axpy_begin) / datetime.timedelta(microseconds=1)

    num_flops = 2 * total_elems

    # element size in bytes (2 for float16)
    elem_bytes = torch.tensor(0, dtype=torch.float16).element_size()

    # approx bytes moved per element: read x, read y, write y
    bytes_per_elem = 3 * elem_bytes
    total_bytes = bytes_per_elem * total_elems

    # microseconds -> seconds
    axpy_time_s = axpy_time / 1e6

    # GFLOPS / TFLOPS and effective bandwidth
    gflops_per_s = (num_flops / 1e9) / axpy_time_s if axpy_time_s > 0 else 0.0
    tflops_per_s = (num_flops / 1e12) / axpy_time_s if axpy_time_s > 0 else 0.0
    bandwidth_GBps = (total_bytes / 1e9) / axpy_time_s if axpy_time_s > 0 else 0.0

    return {
        "size": size,
        "reps": reps,
        "measurement": {
            "vector_generating_time_us": f"{vector_generating_time} microseconds",
            "compute_time_us": f"{axpy_time} microseconds",
            "avg_compute_time_us": f"{axpy_time / reps} microseconds",
            "total_elements": f"{total_elems}",
            "total_flops": f"{num_flops} FLOPs",
            "avg_GFLOPS_per_second": f"{gflops_per_s} GFLOPS/s",
            "avg_TFLOPS_per_second": f"{tflops_per_s} TFLOPS/s",
            "effective_bandwidth_GBps": f"{bandwidth_GBps} GB/s",
        },
    }
