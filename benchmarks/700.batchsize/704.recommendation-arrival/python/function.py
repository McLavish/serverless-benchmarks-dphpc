import datetime
import json
import math
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import storage

client = storage.storage.get_instance()

MODEL_FILE = "dlrm_tiny.pt"
MODEL_CACHE = "/tmp/dlrm_gpu_model"

DEFAULT_EXPERIMENT = {
    "max_batch_size": 16,
    "latency_budget_ms": 500,
    "arrival_rate_rps": 12,
    "simulation_duration_s": 5,
    "max_requests": 0,
    "warmup_runs": 1,
    "report_samples": 3,
    "shuffle": False,
    "seed": 42,
}

_model: Optional[nn.Module] = None
_device = torch.device("cpu")


class TinyDLRM(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embed_dim=8):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.category_emb = nn.Embedding(num_categories, embed_dim)
        in_dim = embed_dim * 3 + 2
        hidden = 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_id, item_id, category_id, dense):
        features = torch.cat(
            [
                self.user_emb(user_id),
                self.item_emb(item_id),
                self.category_emb(category_id),
                dense,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.mlp(features))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA is not available")


def _ensure_model(bucket: str, model_prefix: str) -> Tuple[float, float]:
    global _model, _device

    if _model is not None:
        return 0.0, 0.0

    download_begin = datetime.datetime.now()
    os.makedirs(MODEL_CACHE, exist_ok=True)
    tmp_path = os.path.join("/tmp", f"{uuid.uuid4()}-{MODEL_FILE}")
    client.download(bucket, os.path.join(model_prefix, MODEL_FILE), tmp_path)
    download_end = datetime.datetime.now()

    process_begin = datetime.datetime.now()
    checkpoint = torch.load(tmp_path, map_location="cpu")
    meta = checkpoint["meta"]
    _device = _select_device()
    model = TinyDLRM(
        meta["num_users"], meta["num_items"], meta["num_categories"], meta["embed_dim"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(_device)
    model.eval()
    _model = model
    os.remove(tmp_path)
    process_end = datetime.datetime.now()

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    return download_time, process_time


def _load_requests(path: str, shuffle: bool, seed: int) -> List[dict]:
    with open(path, "r") as f:
        payloads = [json.loads(line) for line in f if line.strip()]
    if not payloads:
        raise RuntimeError("No requests available in dataset.")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(payloads)
    return payloads


def _prepare_batch(requests: Sequence[dict]) -> Tuple[torch.Tensor, ...]:
    user_ids = torch.tensor([req["user_id"] for req in requests], dtype=torch.long, device=_device)
    item_ids = torch.tensor([req["item_id"] for req in requests], dtype=torch.long, device=_device)
    category_ids = torch.tensor(
        [req["category_id"] for req in requests], dtype=torch.long, device=_device
    )
    dense = torch.tensor(
        [req.get("dense", [0.0, 0.0]) for req in requests], dtype=torch.float32, device=_device
    )
    return user_ids, item_ids, category_ids, dense


def _timed_inference(requests: Sequence[dict]) -> Tuple[float, List[float]]:
    inputs = _prepare_batch(requests)
    torch.cuda.synchronize()
    begin = datetime.datetime.now()
    with torch.no_grad():
        scores = _model(*inputs).squeeze(-1).tolist()
    end = datetime.datetime.now()
    torch.cuda.synchronize()
    latency = (end - begin) / datetime.timedelta(microseconds=1)
    return latency, scores


def _format_predictions(requests: Sequence[dict], scores: Sequence[float]) -> List[dict]:
    formatted = []
    for req, score in zip(requests, scores):
        formatted.append(
            {
                "user_id": req["user_id"],
                "item_id": req["item_id"],
                "category_id": req.get("category_id"),
                "score": float(score),
            }
        )
    return formatted


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(values[0])
    if pct >= 1:
        return float(values[-1])
    idx = (len(values) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(values[int(idx)])
    lower_val = values[lower]
    upper_val = values[upper]
    return float(lower_val + (upper_val - lower_val) * (idx - lower))


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    avg = float(sum(sorted_vals) / len(sorted_vals))
    return {
        "count": len(sorted_vals),
        "avg": avg,
        "p50": _percentile(sorted_vals, 0.5),
        "p90": _percentile(sorted_vals, 0.9),
        "p99": _percentile(sorted_vals, 0.99),
        "max": float(sorted_vals[-1]),
    }


def _generate_arrivals(rate_rps: float, duration_s: float, limit: Optional[int], seed: int):
    if rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive")
    rng = random.Random(seed)
    timestamp = 0.0
    arrivals = []
    while timestamp < duration_s:
        inter = rng.expovariate(rate_rps)
        timestamp += inter
        if timestamp > duration_s:
            break
        arrivals.append(timestamp)
        if limit and len(arrivals) >= limit:
            break
    if not arrivals:
        arrivals.append(min(duration_s, 1.0))
    return [int(ts * 1_000_000) for ts in arrivals]


def _make_request(base_requests: List[dict], idx: int, arrival_time: int, identifier: int) -> Dict[str, Any]:
    payload = dict(base_requests[idx % len(base_requests)])
    return {"payload": payload, "arrival_time": arrival_time, "id": identifier}


def _run_warmup(requests: List[dict], runs: int, batch_cap: int) -> float:
    if runs <= 0:
        return 0.0
    usable = min(len(requests), batch_cap)
    if usable == 0:
        raise RuntimeError("Warmup cannot run without any requests.")
    warmup_batch = [requests[i % len(requests)] for i in range(usable)]
    total = 0.0
    for _ in range(runs):
        latency, _ = _timed_inference(warmup_batch)
        total += latency
    return total


def _simulate_batches(
    base_requests: List[dict],
    arrivals_us: List[int],
    max_batch_size: int,
    latency_budget_us: Optional[int],
    report_samples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], float]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    queue: List[Dict[str, Any]] = []
    current_time = 0.0
    arrival_idx = 0
    total_requests = len(arrivals_us)
    req_cursor = 0

    batch_profiles: List[Dict[str, Any]] = []
    request_waits: List[float] = []
    response_times: List[float] = []
    sample_predictions: List[Dict[str, Any]] = []
    batched_compute_time = 0.0
    batch_id = 0
    request_id = 0

    while queue or arrival_idx < total_requests:
        if not queue and arrival_idx < total_requests:
            arrival_time = arrivals_us[arrival_idx]
            current_time = max(current_time, arrival_time)
            queue.append(_make_request(base_requests, req_cursor, arrival_time, request_id))
            req_cursor += 1
            request_id += 1
            arrival_idx += 1
            continue

        next_arrival_time = arrivals_us[arrival_idx] if arrival_idx < total_requests else None
        oldest_arrival = queue[0]["arrival_time"]
        deadline = (
            oldest_arrival + latency_budget_us if latency_budget_us is not None else float("inf")
        )

        should_flush = False
        trigger = None

        if len(queue) >= max_batch_size:
            should_flush = True
            trigger = "capacity"
        elif current_time >= deadline:
            should_flush = True
            trigger = "timeout"
        elif next_arrival_time is None:
            should_flush = True
            trigger = "drain"
        elif next_arrival_time > deadline:
            current_time = max(current_time, deadline)
            should_flush = True
            trigger = "timeout"
        else:
            current_time = max(current_time, next_arrival_time)
            queue.append(_make_request(base_requests, req_cursor, next_arrival_time, request_id))
            req_cursor += 1
            request_id += 1
            arrival_idx += 1
            continue

        batch_size = min(len(queue), max_batch_size)
        batch_requests = [queue.pop(0) for _ in range(batch_size)]
        start_time = max(current_time, batch_requests[-1]["arrival_time"])
        payloads = [entry["payload"] for entry in batch_requests]
        latency, scores = _timed_inference(payloads)
        batched_compute_time += latency
        finish_time = start_time + latency

        waits = [start_time - entry["arrival_time"] for entry in batch_requests]
        responses = [finish_time - entry["arrival_time"] for entry in batch_requests]
        request_waits.extend(waits)
        response_times.extend(responses)

        predictions = _format_predictions(payloads, scores)
        if report_samples > 0 and len(sample_predictions) < report_samples:
            remaining = report_samples - len(sample_predictions)
            sample_predictions.extend(predictions[:remaining])

        batch_profiles.append(
            {
                "batch_id": batch_id,
                "size": batch_size,
                "trigger": trigger,
                "start_time_us": float(start_time),
                "finish_time_us": float(finish_time),
                "inference_time_us": float(latency),
                "wait_min_us": float(min(waits)),
                "wait_max_us": float(max(waits)),
                "wait_avg_us": float(sum(waits) / len(waits)),
            }
        )
        batch_id += 1

        current_time = finish_time
        while arrival_idx < total_requests and arrivals_us[arrival_idx] <= current_time:
            queue.append(_make_request(base_requests, req_cursor, arrivals_us[arrival_idx], request_id))
            req_cursor += 1
            request_id += 1
            arrival_idx += 1

    stats = {
        "total_requests": total_requests,
        "request_wait_summary": _summary(request_waits),
        "request_response_summary": _summary(response_times),
    }
    return batch_profiles, stats, sample_predictions, batched_compute_time


def handler(event):
    bucket = event.get("bucket", {}).get("bucket")
    model_prefix = event.get("bucket", {}).get("model")
    requests_prefix = event.get("bucket", {}).get("requests")
    requests_key = event.get("object", {}).get("requests")

    if not bucket or not model_prefix or not requests_prefix or not requests_key:
        raise ValueError("Missing storage configuration in event payload.")

    download_begin = datetime.datetime.now()
    req_path = os.path.join("/tmp", f"{uuid.uuid4()}-{os.path.basename(requests_key)}")
    client.download(bucket, os.path.join(requests_prefix, requests_key), req_path)
    download_end = datetime.datetime.now()

    model_download_time, model_process_time = _ensure_model(bucket, model_prefix)

    experiment_cfg = dict(DEFAULT_EXPERIMENT)
    experiment_cfg.update(event.get("experiment", {}))

    shuffle = bool(experiment_cfg.get("shuffle", False))
    seed = int(experiment_cfg.get("seed", DEFAULT_EXPERIMENT["seed"]))
    requests = _load_requests(req_path, shuffle, seed)
    os.remove(req_path)

    max_batch_size = int(experiment_cfg.get("max_batch_size", DEFAULT_EXPERIMENT["max_batch_size"]))
    latency_budget_ms = experiment_cfg.get(
        "latency_budget_ms", DEFAULT_EXPERIMENT["latency_budget_ms"]
    )
    latency_budget_us = int(latency_budget_ms * 1000) if latency_budget_ms else None
    arrival_rate = float(experiment_cfg.get("arrival_rate_rps", DEFAULT_EXPERIMENT["arrival_rate_rps"]))
    duration_s = float(
        experiment_cfg.get("simulation_duration_s", DEFAULT_EXPERIMENT["simulation_duration_s"])
    )
    max_requests = experiment_cfg.get("max_requests", DEFAULT_EXPERIMENT["max_requests"])
    max_requests = int(max_requests) if max_requests else None
    arrivals_us = _generate_arrivals(arrival_rate, duration_s, max_requests, seed)

    warmup_runs = int(experiment_cfg.get("warmup_runs", DEFAULT_EXPERIMENT["warmup_runs"]))
    warmup_time = _run_warmup(requests, warmup_runs, max_batch_size)

    report_samples = int(experiment_cfg.get("report_samples", DEFAULT_EXPERIMENT["report_samples"]))

    profiling_begin = datetime.datetime.now()
    batch_profiles, stats, sample_predictions, batched_compute_time = _simulate_batches(
        requests, arrivals_us, max_batch_size, latency_budget_us, report_samples
    )
    profiling_end = datetime.datetime.now()

    first_arrival = arrivals_us[0]
    last_completion = batch_profiles[-1]["finish_time_us"] if batch_profiles else first_arrival
    simulated_span = max(last_completion - first_arrival, 1.0)
    stats["simulation_span_us"] = simulated_span
    throughput = stats["total_requests"] / (simulated_span / 1_000_000.0)

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    profiling_time = (profiling_end - profiling_begin) / datetime.timedelta(microseconds=1)

    return {
        "result": {
            "simulation": {
                "arrival_rate_rps": arrival_rate,
                "simulation_duration_s": duration_s,
                "latency_budget_ms": latency_budget_ms,
                "max_batch_size": max_batch_size,
                "throughput_rps": throughput,
            },
            "requests": stats,
            "batch_profiles": batch_profiles,
            "sample_predictions": sample_predictions,
        },
        "measurement": {
            "download_time": download_time + model_download_time,
            "compute_time": profiling_time + model_process_time + warmup_time,
            "batched_compute_time": batched_compute_time + warmup_time,
            "model_time": model_process_time,
            "model_download_time": model_download_time,
            "warmup_time": warmup_time,
        },
    }
