# experiment/gpu_app_metrics.py

import os
import re
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch

import json


def unwrap_result(out):
    """
    convert sebs.py wrapped format into handler output dict.
    supports:
      - str: JSON string
      - {"statusCode": ..., "body": "..."}  (HTTP trigger)
      - {"result": {...}}
      - {"output": {...}}   (local.HTTPTrigger)
    """

    # 1) stringify JSON
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            return out

    # 2) HTTP trigger: {"statusCode": ..., "body": "..."}
    if isinstance(out, dict) and "body" in out:
        body = out["body"]
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                pass
        out = body

    # 3) sometimes wrapped as {"result": {...}}
    if isinstance(out, dict) and "result" in out and isinstance(out["result"], dict):
        out = out["result"]

    # 4) local HTTPTrigger: {"output": {...}, "stats": ..., ...}
    if isinstance(out, dict) and "output" in out and isinstance(out["output"], dict):
        out = out["output"]

    return out


# -------- global --------
RUNS = 50


OUTPUT_DIR = Path(".")

# size / iters
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
COMPUTE_ITERS = 50

CHANNEL_FLOW_SIZES = [
    ({"ny": 41, "nx": 41, "nit": 50, "rho": 1.0, "nu": 0.1, "F": 1.0}, "41x41,nit=50"),
    ({"ny": 81, "nx": 81, "nit": 50, "rho": 1.0, "nu": 0.1, "F": 1.0}, "81x81,nit=50"),
    ({"ny": 161, "nx": 161, "nit": 50, "rho": 1.0, "nu": 0.1, "F": 1.0}, "161x161,nit=50"),
]


# -------- utility --------


def sanitize_name(name: str) -> str:
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_")


def plot_box(exp_name, metric_name, size_labels, data_lists, ylabel, logy=False):

    fig, ax = plt.subplots(figsize=(6.5, 2.6))

    bp = ax.boxplot(
        data_lists,
        tick_labels=size_labels,
        showmeans=False,
        showfliers=True,
        patch_artist=True,
    )

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors[: len(data_lists)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel(ylabel)
    ax.set_title(f"{exp_name} – {metric_name}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    if logy:
        ax.set_yscale("log")

    fig.tight_layout()
    fname = OUTPUT_DIR / f"{sanitize_name(exp_name)}_{sanitize_name(metric_name)}_box.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def theoretical_mem_bw_gbps(device_index: int = 0) -> float | None:
    prop = torch.cuda.get_device_properties(device_index)

    clock_khz = getattr(prop, "memory_clock_rate", None)
    bus_width_bits = getattr(prop, "memory_bus_width", None)

    # some PyTorch versions / devices may not have these attributes
    if clock_khz is None or bus_width_bits is None:
        print(
            "[gpu-app-metrics] memory_clock_rate / memory_bus_width not available, "
            "skip BW utilization metric."
        )
        return None

    clock_hz = clock_khz * 1e3  # kHz -> Hz
    bus_bytes = bus_width_bits / 8.0  # bits -> bytes
    bw = 2.0 * clock_hz * bus_bytes / 1e9  # GB/s
    return bw


# -------- host-device-copy --------


def run_host_device_copy(invoke: Callable[[Dict[str, Any]], Dict[str, Any]]):
    metrics = {
        "PCIe H2D BW [GB/s]": {label: [] for _, label in HD_COPY_SIZES},
        "PCIe D2H BW [GB/s]": {label: [] for _, label in HD_COPY_SIZES},
        "H2D latency [ms]": {label: [] for _, label in HD_COPY_SIZES},
        "D2H latency [ms]": {label: [] for _, label in HD_COPY_SIZES},
        "GPU copy utilization": {label: [] for _, label in HD_COPY_SIZES},
    }

    # warmup
    invoke({"size": HD_COPY_SIZES[0][0], "iters": HD_COPY_ITERS})

    for size, label in HD_COPY_SIZES:
        for _ in range(RUNS):
            raw = invoke({"size": size, "iters": HD_COPY_ITERS})
            out = unwrap_result(raw)
            if "H2D_effective_BW" not in out:

                print("[gpu-app-metrics] unexpected host-device-copy output:", out)
                raise RuntimeError(
                    f"host-device-copy result missing H2D_effective_BW, got keys={list(out.keys())}"
                )

            h2d_bw = float(out["H2D_effective_BW"].split()[0])
            d2h_bw = float(out["D2H_effective_BW"].split()[0])
            h2d_avg = float(out["H2D_avg"].split()[0])
            d2h_avg = float(out["D2H_avg"].split()[0])

            m = out.get("measurement", {})
            h2d_total = float(m.get("H2D_total_time_ms", 0.0))
            d2h_total = float(m.get("D2H_total_time_ms", 0.0))
            gen_time = float(m.get("generate_time_ms", 0.0))
            total_time = h2d_total + d2h_total + gen_time
            util = 0.0 if total_time <= 0 else (h2d_total + d2h_total) / total_time

            metrics["PCIe H2D BW [GB/s]"][label].append(h2d_bw)
            metrics["PCIe D2H BW [GB/s]"][label].append(d2h_bw)
            metrics["H2D latency [ms]"][label].append(h2d_avg)
            metrics["D2H latency [ms]"][label].append(d2h_avg)
            metrics["GPU copy utilization"][label].append(util)

    return metrics


# -------- vector-add --------


def run_vector_add(invoker):
    metrics = {
        "Memory throughput [GB/s]": {label: [] for _, label in VECADD_SIZES},
        "Memory BW utilization [%]": {label: [] for _, label in VECADD_SIZES},
        "Per-iter latency [ms]": {label: [] for _, label in VECADD_SIZES},
    }

    # warmup
    first_size = VECADD_SIZES[0][0]
    invoker({"size": first_size, "iters": VECADD_ITERS})

    peak_bw = theoretical_mem_bw_gbps(0)  # maybe None

    for size, label in VECADD_SIZES:
        for _ in range(RUNS):
            raw = invoker({"size": size, "iters": VECADD_ITERS})
            out = unwrap_result(raw)

            payload = out.get("output", out)

            bw = float(payload["effective BW"].split()[0])
            avg_ms = float(payload["avg"].split()[0])

            if peak_bw and peak_bw > 0:
                util = 100.0 * bw / peak_bw
            else:
                util = 0.0

            metrics["Memory throughput [GB/s]"][label].append(bw)
            metrics["Memory BW utilization [%]"][label].append(util)
            metrics["Per-iter latency [ms]"][label].append(avg_ms)

    return metrics


# -------- compute (JAX) --------


def run_compute(invoke: Callable[[Dict[str, Any]], Dict[str, Any]]):

    metrics = {
        "GFLOP/s": {label: [] for _, label in COMPUTE_SIZES},
        "Arithmetic intensity [FLOP/byte]": {label: [] for _, label in COMPUTE_SIZES},
        "compute time [ms]": {label: [] for _, label in COMPUTE_SIZES},
    }

    # warmup， JIT
    (M0, N0), _ = COMPUTE_SIZES[0]
    invoke({"size": {"M": M0, "N": N0}, "iters": COMPUTE_ITERS})

    for (shape, label) in COMPUTE_SIZES:
        M, N = shape
        num_elems = M * N

        # compute flops:
        flops_per_elem = 4.0
        flops_total = flops_per_elem * num_elems * COMPUTE_ITERS

        # compute arithmetic intensity:
        bytes_per_elem = 3.0 * 4.0
        intensity = flops_per_elem / bytes_per_elem

        size_arg = {"M": M, "N": N}

        for _ in range(RUNS):
            raw = invoke({"size": size_arg, "iters": COMPUTE_ITERS})
            out = unwrap_result(raw)
            m = out.get("measurement", {})
            t_ms = float(m.get("compute_time", 0.0))
            t_s = t_ms / 1000.0 if t_ms > 0 else 0.0
            gflops = 0.0 if t_s <= 0 else (flops_total / 1e9) / t_s

            metrics["GFLOP/s"][label].append(gflops)
            metrics["Arithmetic intensity [FLOP/byte]"][label].append(intensity)
            metrics["compute time [ms]"][label].append(t_ms)

    return metrics


# -------- channel_flow (JAX CFD) --------


def run_channel_flow(invoke: Callable[[Dict[str, Any]], Dict[str, Any]]):

    metrics = {
        "compute time [ms]": {label: [] for _, label in CHANNEL_FLOW_SIZES},
    }

    # warmup
    invoke({"size": CHANNEL_FLOW_SIZES[0][0]})

    for size_dict, label in CHANNEL_FLOW_SIZES:
        for _ in range(RUNS):
            raw = invoke({"size": size_dict})
            out = unwrap_result(raw)
            m = out.get("measurement", {})
            t_ms = float(m.get("compute_time", 0.0))
            metrics["compute time [ms]"][label].append(t_ms)

    return metrics


# -------- interact with SeBS  --------


def run_all(invokers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]], out_dir: str = "."):
    """
    invokers: dict
      key: benchmark
           - "host-device-copy"
           - "vector-add"
           - "compute"
           - "channel-flow"
      value:  event -> dict
    """
    global OUTPUT_DIR
    OUTPUT_DIR = Path(out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # host-device-copy
    if "host-device-copy" in invokers:
        print("\n=== host-device-copy (app-level) ===")
        hd_metrics = run_host_device_copy(invokers["host-device-copy"])
        for metric_name, size_dict in hd_metrics.items():
            labels = list(size_dict.keys())
            data_lists = [size_dict[l] for l in labels]
            logy = "latency" in metric_name
            plot_box("host-device-copy", metric_name, labels, data_lists, metric_name, logy=logy)

    # vector-add
    if "vector-add" in invokers:
        print("\n=== vector-add (app-level) ===")
        vec_metrics = run_vector_add(invokers["vector-add"])
        for metric_name, size_dict in vec_metrics.items():
            labels = list(size_dict.keys())
            data_lists = [size_dict[l] for l in labels]
            plot_box("vector-add", metric_name, labels, data_lists, metric_name, logy=False)

    # compute
    if "compute" in invokers:
        print("\n=== compute (app-level) ===")
        comp_metrics = run_compute(invokers["compute"])
        for metric_name, size_dict in comp_metrics.items():
            labels = list(size_dict.keys())
            data_lists = [size_dict[l] for l in labels]
            logy = "time" in metric_name
            plot_box("compute", metric_name, labels, data_lists, metric_name, logy=logy)

    # channel_flow
    if "channel-flow" in invokers:
        print("\n=== channel_flow (app-level) ===")
        ch_metrics = run_channel_flow(invokers["channel-flow"])
        for metric_name, size_dict in ch_metrics.items():
            labels = list(size_dict.keys())
            data_lists = [size_dict[l] for l in labels]
            plot_box("channel_flow", metric_name, labels, data_lists, metric_name, logy=True)

    print("\nAll app-level figures saved in", OUTPUT_DIR)


# reservered for direct execution

if __name__ == "__main__":
    print("This module is intended to be used via SeBS (experiment gpu-app-metrics).")
