import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import re
import sys
import shutil
import subprocess
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same sizes as single_run_target.py
HD_COPY_SIZES = [
    (1 << 22, "2^22 elems"),
    (1 << 24, "2^24 elems"),
    (1 << 26, "2^26 elems"),
]

VECADD_SIZES = [
    (1 << 22, "2^22 elems"),
    (1 << 24, "2^24 elems"),
    (1 << 26, "2^26 elems"),
]

COMPUTE_SIZES = [
    ((1024, 1024), "1024x1024"),
    ((2048, 2048), "2048x2048"),
    ((4096, 4096), "4096x4096"),
]

CHANNEL_FLOW_SIZES = [
    ({"ny": 61,  "nx": 61,  "nit": 5, "rho": 1.0, "nu": 0.1, "F": 1.0}, "61x61,nit=5"),
    ({"ny": 121,  "nx": 121,  "nit": 10, "rho": 1.0, "nu": 0.1, "F": 1.0}, "121x121,nit=10"),
    ({"ny": 201, "nx": 201, "nit": 20, "rho": 1.0, "nu": 0.1, "F": 1.0}, "201x201,nit=20"),
]

# Nsight Compute metrics to collect
NCU_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",    # Achieved occupancy (%)
    "sm__inst_issued.avg.per_cycle_active",                 # IPC (inst/cycle)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",     # SM utilization (%)
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",   # DRAM throughput (% of peak)
    "lts__t_sectors_hit_rate.pct",                          # L2 hit rate (%)
]

PROF_RUNS = 1  # how many times to profile each (exp, size)

# ---- auto-detect tools ----
NCU_EXE = shutil.which("ncu")
if NCU_EXE is None:
    raise RuntimeError(
        "Could not find 'ncu' in PATH. In this terminal, run 'ncu --version' first.\n"
        "If that fails, add Nsight Compute's bin directory to PATH, or set NCU_EXE manually."
    )

PYTHON_EXE = sys.executable  # the python that runs this script

# Ensure we find single_run_target.py in the same directory as this script
HERE = Path(__file__).resolve().parent
SINGLE_RUN_SCRIPT = HERE / "gpu_hw_singlerun.py"
if not SINGLE_RUN_SCRIPT.exists():
    raise RuntimeError(
        f"gpu_hw_singlerun.py not found at {SINGLE_RUN_SCRIPT}. "
        "Place this script next to profile_hw_metrics_with_ncu.py."
    )

# === 新增：统一输出目录 gpu_hw_output 在脚本所在目录下 ===
OUTPUT_DIR = HERE / "gpu_hw_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[gpu-hw-metrics] outputs will be saved under {OUTPUT_DIR}")


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
    # ==== 修改：图像保存到 OUTPUT_DIR ====
    fname = OUTPUT_DIR / f"{sanitize_name(exp_name)}_{sanitize_name(metric_name)}_HW_box.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def ensure_ncu_available():
    if shutil.which("ncu") is None:
        raise RuntimeError(
            "Nsight Compute CLI 'ncu' not found in PATH. "
            "Please install Nsight Compute and ensure 'ncu' is available."
        )


def parse_ncu_csv(stdout: str, wanted_metrics: set[str]) -> dict[str, float]:
    """
    Parse Nsight Compute CSV output.

    - Drop non-CSV lines (==PROF== ...)
    - Strip possible UTF-8 BOM
    - Use DictReader so we can access "Metric Name" / "Metric Value"
    - Aggregate metrics over all kernels; last occurrence wins
    """
    lines = stdout.splitlines()

    csv_lines: list[str] = []
    for ln in lines:
        # strip BOM if present
        ln = ln.lstrip("\ufeff")
        # keep only lines that look like CSV rows
        if ln.startswith('"'):
            csv_lines.append(ln)

    if not csv_lines:
        return {}

    buf = io.StringIO("\n".join(csv_lines))
    reader = csv.DictReader(buf)

    # sanity check header
    if not reader.fieldnames or "Metric Name" not in reader.fieldnames or "Metric Value" not in reader.fieldnames:
        return {}

    result: dict[str, float] = {}

    for row in reader:
        mname = row.get("Metric Name")
        if mname not in wanted_metrics:
            continue

        v_str = row.get("Metric Value") or ""
        v_str = v_str.replace("%", "").replace(",", "")

        try:
            val = float(v_str)
        except ValueError:
            # e.g. "n/a"
            continue

        # last occurrence wins
        result[mname] = val

    return result


def profile_one(exp: str, size_index: int) -> list[dict[str, float]]:
    """
    Run Nsight Compute PROF_RUNS times for a given (exp, size_index),
    return list of metric dicts for each successful run.
    Also dumps raw ncu stdout to gpu_hw_output/ncu_debug_...txt
    for debugging.
    """
    wanted = set(NCU_METRICS)
    results: list[dict[str, float]] = []

    for r in range(PROF_RUNS):
        cmd = [
            NCU_EXE,
            "--target-processes", "all",
            "--metrics", ",".join(NCU_METRICS),
            "--csv",
            PYTHON_EXE,
            str(SINGLE_RUN_SCRIPT),
            "--exp", exp,
            "--size_index", str(size_index),
        ]
        if exp == "channel-flow":
            cmd = [
                NCU_EXE,
                "--launch-skip", "120",
                "--launch-count", "60",
                "--target-processes", "all",
                "--metrics", ",".join(NCU_METRICS),
                "--csv",
                PYTHON_EXE,
                str(SINGLE_RUN_SCRIPT),
                "--exp", exp,
                "--size_index", str(size_index),
            ]
        print(f"  [ncu] {exp}, size_index={size_index}, run {r+1}/{PROF_RUNS}")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        # ==== 修改：debug 文件写到 OUTPUT_DIR ====
        debug_fname = OUTPUT_DIR / f"ncu_debug_{sanitize_name(exp)}_size{size_index}_run{r+1}.txt"
        with open(debug_fname, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(proc.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(proc.stderr)
        print(f"    raw ncu output saved to {debug_fname}")

        if proc.returncode != 0:
            print("    WARNING: ncu returned non-zero code "
                  f"(exit {proc.returncode}), see debug file.")
            continue

        metrics = parse_ncu_csv(proc.stdout, wanted)
        if not metrics:
            print("    WARNING: could not parse metrics from ncu CSV output "
                  f"(see {debug_fname})")
        else:
            results.append(metrics)

    return results


def collect_hw_metrics():
    """
    Collect HW metrics for all experiments and sizes using Nsight Compute.
    Returns a nested dict:

      hw_results[exp][metric_name][size_label] = [values...]
    """
    hw_results: dict[str, dict[str, dict[str, list[float]]]] = {}

    experiments = {
        # "channel-flow": CHANNEL_FLOW_SIZES,
        # "host-device-copy": HD_COPY_SIZES,
        "vector-add": VECADD_SIZES,
        # "compute": COMPUTE_SIZES,
    }

    for exp, sizes in experiments.items():
        print(f"\n=== Profiling HW metrics for {exp} ===")
        hw_results[exp] = {m: {label: [] for _, label in sizes} for m in NCU_METRICS}

        for idx, (_, label) in enumerate(sizes):
            print(f" Profiling size {label} (index {idx})...")
            run_metric_list = profile_one(exp, idx)
            for metrics in run_metric_list:
                for mname in NCU_METRICS:
                    if mname in metrics:
                        hw_results[exp][mname][label].append(metrics[mname])

    return hw_results


def main():
    ensure_ncu_available()
    if not SINGLE_RUN_SCRIPT.exists():
        raise RuntimeError(
            f"{SINGLE_RUN_SCRIPT} not found. "
            "Please place this script in the same directory as gpu_hw_singlerun.py."
        )

    hw_results = collect_hw_metrics()

    readable_names = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy [%]",
        "sm__inst_issued.avg.per_cycle_active": "IPC [inst/cycle]",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput [% of peak]",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput [% of peak]",
        "lts__t_sectors_hit_rate.pct": "L2 Hit Rate [%]",
    }

    for exp, metrics_dict in hw_results.items():
        for metric_name, size_dict in metrics_dict.items():
            labels = list(size_dict.keys())
            data_lists = [size_dict[l] for l in labels]
            if all(len(lst) == 0 for lst in data_lists):
                print(f"Skipping {exp} / {metric_name} (no data).")
                continue

            display_name = readable_names.get(metric_name, metric_name)
            plot_box(
                exp_name=f"{exp} (HW)",
                metric_name=display_name,
                size_labels=labels,
                data_lists=data_lists,
                ylabel=display_name,
                logy=False,
            )

    print("\nHardware-level metric figures saved under", OUTPUT_DIR)


if __name__ == "__main__":
    main()
