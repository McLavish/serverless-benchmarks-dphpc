import datetime
import json
import os
import random
import statistics
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import storage

client = storage.storage.get_instance()

MODEL_FILE = "dlrm_tiny.pt"
MODEL_CACHE = "/tmp/dlrm_gpu_model"

DEFAULT_EXPERIMENT = {
    "batch_sizes": [1, 2, 4, 8],
    "request_limit": 128,
    "warmup_runs": 1,
    "repetitions": 2,
    "shuffle": False,
    "seed": 42,
    "report_predictions": 3,
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


def _load_requests(path: str, limit: Optional[int], shuffle: bool, seed: int) -> List[dict]:
    with open(path, "r") as f:
        payloads = [json.loads(line) for line in f if line.strip()]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(payloads)
    if limit and limit > 0:
        payloads = payloads[:limit]
    if not payloads:
        raise RuntimeError("No requests available for profiling.")
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


def _profile_batch_size(
    requests: Sequence[dict],
    batch_size: int,
    warmup_runs: int,
    repetitions: int,
    report_predictions: int,
) -> Tuple[Dict[str, Any], float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    usable = (len(requests) // batch_size) * batch_size
    if usable == 0:
        raise ValueError(f"Insufficient requests ({len(requests)}) for batch_size={batch_size}")
    batches = [requests[i : i + batch_size] for i in range(0, usable, batch_size)]

    warmup = max(0, int(warmup_runs))
    for i in range(warmup):
        _timed_inference(batches[i % len(batches)])

    timings = []
    total_batches = 0
    sample_predictions: Optional[List[dict]] = None
    reps = max(1, int(repetitions))
    for _ in range(reps):
        for chunk in batches:
            latency, scores = _timed_inference(chunk)
            timings.append(latency)
            total_batches += 1
            if sample_predictions is None and report_predictions > 0:
                sample_predictions = _format_predictions(chunk, scores)[:report_predictions]

    total_latency = float(sum(timings))
    samples_processed = total_batches * batch_size
    throughput = samples_processed / total_latency * 1e6 if total_latency > 0 else 0.0

    profile: Dict[str, Any] = {
        "batch_size": batch_size,
        "batches": total_batches,
        "samples_processed": samples_processed,
        "total_latency_us": total_latency,
        "min_latency_us": float(min(timings)),
        "max_latency_us": float(max(timings)),
        "mean_latency_us": float(statistics.mean(timings)),
        "median_latency_us": float(statistics.median(timings)),
        "samples_per_second": throughput,
    }
    if sample_predictions:
        profile["sample_predictions"] = sample_predictions
    return profile, total_latency


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
    request_limit = experiment_cfg.get("request_limit", DEFAULT_EXPERIMENT["request_limit"])
    requests = _load_requests(req_path, request_limit, shuffle, seed)
    os.remove(req_path)

    batch_sizes = experiment_cfg.get("batch_sizes", DEFAULT_EXPERIMENT["batch_sizes"])
    if not batch_sizes:
        raise ValueError("At least one batch size must be provided.")
    batch_sizes = [int(size) for size in batch_sizes]

    warmup_runs = int(experiment_cfg.get("warmup_runs", DEFAULT_EXPERIMENT["warmup_runs"]))
    repetitions = int(experiment_cfg.get("repetitions", DEFAULT_EXPERIMENT["repetitions"]))
    report_predictions = int(
        experiment_cfg.get("report_predictions", DEFAULT_EXPERIMENT["report_predictions"])
    )

    profiling_begin = datetime.datetime.now()
    profiles = []
    total_latency = 0.0
    for batch_size in batch_sizes:
        profile, latency = _profile_batch_size(
            requests, batch_size, warmup_runs, repetitions, report_predictions
        )
        profiles.append(profile)
        total_latency += latency
    profiling_end = datetime.datetime.now()

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    profiling_time = (profiling_end - profiling_begin) / datetime.timedelta(microseconds=1)

    return {
        "result": {
            "requests_used": len(requests),
            "batch_profiles": profiles,
        },
        "measurement": {
            "download_time": download_time + model_download_time,
            "compute_time": profiling_time + model_process_time,
            "batched_compute_time": total_latency,
            "model_time": model_process_time,
            "model_download_time": model_download_time,
        },
    }
