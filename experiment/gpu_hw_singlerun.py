import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import importlib.util
from pathlib import Path

# ====== TODO: set these paths to your actual function.py ======
HOST_DEVICE_COPY_PATH = Path("../benchmarks/000.microbenchmarks/0xx.host-device-copy/python/function.py")
VECTOR_ADD_PATH       = Path("../benchmarks/000.microbenchmarks/0xx.vector-add/python/function.py")
COMPUTE_PATH          = Path("compute.py")
CHANNEL_FLOW_PATH     = Path("channel_flow.py")

# ===============================================================

# Sizes must match what you use in your main benchmark script
HD_COPY_SIZES = [
    (1 << 22, "2^22 elems"),
    (1 << 24, "2^24 elems"),
    (1 << 26, "2^26 elems"),
]
HD_COPY_ITERS = 10

VECADD_SIZES = [
    (1 << 22, "2^22 elems"),
    (1 << 24, "2^24 elems"),
    (1 << 26, "2^26 elems"),
]
VECADD_ITERS = 10

COMPUTE_SIZES = [
    ((1024, 1024), "1024x1024"),
    ((2048, 2048), "2048x2048"),
    ((4096, 4096), "4096x4096"),
]

CHANNEL_FLOW_SIZES = [
    ({"ny": 61,  "nx": 61,  "nit": 5, "rho": 1.0, "nu": 0.1, "F": 1.0}, "61x61,nit=5"),
    ({"ny": 121,  "nx": 121,  "nit": 10, "rho": 1.0, "nu": 0.1, "F": 1.0}, "121x121,nit=5"),
    ({"ny": 201, "nx": 201, "nit": 20, "rho": 1.0, "nu": 0.1, "F": 1.0}, "201x201,nit=5"),
]


def load_handler_from_file(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    module_name = "mod_single_" + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "handler"):
        raise AttributeError(f"{path} does not define a `handler(event)` function")
    return module.handler


def main():
    parser = argparse.ArgumentParser(
        description="Single run target for Nsight Compute profiling."
    )
    parser.add_argument(
        "--exp",
        required=True,
        choices=["host-device-copy", "vector-add", "compute", "channel-flow"],
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--size_index",
        type=int,
        required=True,
        help="Index of size configuration (0, 1, or 2).",
    )
    args = parser.parse_args()

    if args.exp == "host-device-copy":
        handler = load_handler_from_file(HOST_DEVICE_COPY_PATH)
        size = HD_COPY_SIZES[args.size_index][0]
        event = {"size": size, "iters": HD_COPY_ITERS}
        handler(event)

    elif args.exp == "vector-add":
        handler = load_handler_from_file(VECTOR_ADD_PATH)
        size = VECADD_SIZES[args.size_index][0]
        event = {"size": size, "iters": VECADD_ITERS}
        handler(event)

    elif args.exp == "compute":
        handler = load_handler_from_file(COMPUTE_PATH)
        M, N = COMPUTE_SIZES[args.size_index][0]
        event = {"size": {"M": M, "N": N}}
        handler(event)

    elif args.exp == "channel-flow":
        handler = load_handler_from_file(CHANNEL_FLOW_PATH)
        size_dict = CHANNEL_FLOW_SIZES[args.size_index][0]
        event = {"size": size_dict}
        handler(event)

    else:
        raise ValueError(f"Unknown exp: {args.exp}")


if __name__ == "__main__":
    main()