import datetime
import json
import math
import os
import random
import shutil
import tarfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import storage

client = storage.storage.get_instance()

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
class_idx = json.load(open(os.path.join(SCRIPT_DIR, "imagenet_class_index.json"), "r"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

MODEL_ARCHIVE = "resnet50.tar.gz"
MODEL_DIRECTORY = "/tmp/image_classification_model"
MODEL_SUBDIR = "resnet50"

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

_session: Optional[ort.InferenceSession] = None
_session_input: Optional[str] = None
_session_output: Optional[str] = None
_cached_model_key: Optional[str] = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_model(bucket: str, model_prefix: str, model_key: Optional[str]) -> Tuple[float, float]:
    global _session, _session_input, _session_output, _cached_model_key

    effective_model_key = model_key or MODEL_ARCHIVE
    model_download_begin = datetime.datetime.now()
    model_download_end = model_download_begin

    if _session is None or _cached_model_key != effective_model_key:
        archive_basename = os.path.basename(effective_model_key)
        archive_path = os.path.join("/tmp", f"{uuid.uuid4()}-{archive_basename}")
        model_dir = os.path.join(MODEL_DIRECTORY, MODEL_SUBDIR)

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(MODEL_DIRECTORY, exist_ok=True)

        client.download(bucket, os.path.join(model_prefix, effective_model_key), archive_path)
        model_download_end = datetime.datetime.now()

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(MODEL_DIRECTORY)
        os.remove(archive_path)

        model_process_begin = datetime.datetime.now()
        onnx_path = os.path.join(model_dir, "model.onnx")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable (providers: {available})")

        _session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        _session_input = _session.get_inputs()[0].name
        _session_output = _session.get_outputs()[0].name
        _cached_model_key = effective_model_key
        model_process_end = datetime.datetime.now()
    else:
        model_process_begin = datetime.datetime.now()
        model_process_end = model_process_begin

    model_download_time = (model_download_end - model_download_begin) / datetime.timedelta(
        microseconds=1
    )
    model_process_time = (model_process_end - model_process_begin) / datetime.timedelta(
        microseconds=1
    )
    return model_download_time, model_process_time


def _resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(round(size * height / width))
    else:
        new_height = size
        new_width = int(round(size * width / height))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((new_width, new_height), resample=resample)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, int(round((width - size) / 2)))
    top = max(0, int(round((height - size) / 2)))
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))


def _prepare_tensor(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = _resize_shorter_side(image, 256)
    image = _center_crop(image, 224)

    np_image = np.asarray(image).astype(np.float32) / 255.0
    np_image = (np_image - _MEAN) / _STD
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image


def _load_images_list(entries: Sequence[Sequence[str]], shuffle: bool, seed: int) -> List[Tuple[str, str]]:
    images = list(entries)
    if not images:
        raise RuntimeError("No image entries supplied.")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(images)
    return images


def _download_image(bucket: str, prefix: str, img: str) -> Tuple[str, float]:
    download_path = os.path.join("/tmp", f"{uuid.uuid4()}-{os.path.basename(img)}")
    begin = datetime.datetime.now()
    client.download(bucket, os.path.join(prefix, img), download_path)
    end = datetime.datetime.now()
    download_time = (end - begin) / datetime.timedelta(microseconds=1)
    return download_path, download_time


def _tensor_for_entry(
    entries: List[Tuple[str, str]], idx: int, bucket: str, prefix: str
) -> Tuple[np.ndarray, float]:
    img, _label = entries[idx % len(entries)]
    path, download_time = _download_image(bucket, prefix, img)
    tensor = _prepare_tensor(path)
    os.remove(path)
    return tensor, download_time


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _format_predictions(logits: np.ndarray) -> List[Dict[str, Any]]:
    probs = _softmax(logits)
    formatted = []
    for prob in probs:
        top1_idx = int(np.argmax(prob))
        top5_idx = np.argsort(prob)[::-1][:5].tolist()
        formatted.append(
            {
                "label": idx2label[top1_idx],
                "confidence": float(prob[top1_idx]),
                "top5": [idx2label[idx] for idx in top5_idx],
            }
        )
    return formatted


def _timed_inference(batch: np.ndarray) -> Tuple[float, np.ndarray]:
    assert _session is not None and _session_input is not None and _session_output is not None
    begin = datetime.datetime.now()
    outputs = _session.run([_session_output], {_session_input: batch})
    end = datetime.datetime.now()
    latency = (end - begin) / datetime.timedelta(microseconds=1)
    return latency, outputs[0]


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


def _make_request(
    entries: List[Tuple[str, str]],
    idx: int,
    arrival_time: int,
    bucket: str,
    prefix: str,
) -> Dict[str, Any]:
    tensor, download_time = _tensor_for_entry(entries, idx, bucket, prefix)
    return {
        "tensor": tensor,
        "arrival_time": arrival_time,
        "id": idx,
        "download_time_us": download_time,
    }


def _run_warmup(
    entries: List[Tuple[str, str]],
    runs: int,
    batch_cap: int,
    bucket: str,
    prefix: str,
) -> Tuple[float, float]:
    if runs <= 0:
        return 0.0, 0.0
    usable = min(len(entries), batch_cap)
    if usable == 0:
        raise RuntimeError("Warmup cannot run without tensors.")
    tensors = []
    download_time = 0.0
    for idx in range(usable):
        tensor, dl = _tensor_for_entry(entries, idx, bucket, prefix)
        tensors.append(tensor)
        download_time += dl
    batch = np.stack(tensors, axis=0)
    total = 0.0
    for _ in range(runs):
        latency, _ = _timed_inference(batch)
        total += latency
    return total, download_time


def _simulate_batches(
    entries: List[Tuple[str, str]],
    bucket: str,
    prefix: str,
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
    entry_cursor = 0

    batch_profiles: List[Dict[str, Any]] = []
    request_waits: List[float] = []
    response_times: List[float] = []
    sample_predictions: List[Dict[str, Any]] = []
    batched_compute_time = 0.0
    batch_id = 0

    total_download_time = 0.0
    while queue or arrival_idx < total_requests:
        if not queue and arrival_idx < total_requests:
            arrival_time = arrivals_us[arrival_idx]
            current_time = max(current_time, arrival_time)
            request = _make_request(entries, entry_cursor, arrival_time, bucket, prefix)
            total_download_time += request["download_time_us"]
            queue.append(request)
            entry_cursor += 1
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
            request = _make_request(entries, entry_cursor, next_arrival_time, bucket, prefix)
            total_download_time += request["download_time_us"]
            queue.append(request)
            entry_cursor += 1
            arrival_idx += 1
            continue

        batch_size = min(len(queue), max_batch_size)
        batch_requests = [queue.pop(0) for _ in range(batch_size)]
        start_time = max(current_time, batch_requests[-1]["arrival_time"])
        tensors = [req["tensor"] for req in batch_requests]
        batch = np.stack(tensors, axis=0)
        latency, logits = _timed_inference(batch)
        batched_compute_time += latency
        finish_time = start_time + latency

        waits = [start_time - req["arrival_time"] for req in batch_requests]
        responses = [finish_time - req["arrival_time"] for req in batch_requests]
        request_waits.extend(waits)
        response_times.extend(responses)

        predictions = _format_predictions(logits)
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
            request = _make_request(entries, entry_cursor, arrivals_us[arrival_idx], bucket, prefix)
            total_download_time += request["download_time_us"]
            queue.append(request)
            entry_cursor += 1
            arrival_idx += 1

    stats = {
        "total_requests": total_requests,
        "request_wait_summary": _summary(request_waits),
        "request_response_summary": _summary(response_times),
    }
    return batch_profiles, stats, sample_predictions, batched_compute_time, total_download_time


def handler(event):
    bucket = event.get("bucket", {}).get("bucket")
    model_prefix = event.get("bucket", {}).get("model")
    images_prefix = event.get("bucket", {}).get("images")
    entries = event.get("object", {}).get("images", [])
    model_key = event.get("object", {}).get("model")

    if not bucket or not model_prefix or not images_prefix or not entries:
        raise ValueError("Missing storage configuration or image list in event payload.")

    experiment_cfg = dict(DEFAULT_EXPERIMENT)
    experiment_cfg.update(event.get("experiment", {}))

    shuffle = bool(experiment_cfg.get("shuffle", False))
    seed = int(experiment_cfg.get("seed", DEFAULT_EXPERIMENT["seed"]))
    image_entries = _load_images_list(entries, shuffle, seed)

    model_download_time, model_process_time = _ensure_model(bucket, model_prefix, model_key)

    arrival_rate = float(experiment_cfg.get("arrival_rate_rps", DEFAULT_EXPERIMENT["arrival_rate_rps"]))
    duration_s = float(
        experiment_cfg.get("simulation_duration_s", DEFAULT_EXPERIMENT["simulation_duration_s"])
    )
    max_requests = experiment_cfg.get("max_requests", DEFAULT_EXPERIMENT["max_requests"])
    max_requests = int(max_requests) if max_requests else None
    arrivals_us = _generate_arrivals(arrival_rate, duration_s, max_requests, seed)

    max_batch_size = int(experiment_cfg.get("max_batch_size", DEFAULT_EXPERIMENT["max_batch_size"]))
    latency_budget_ms = experiment_cfg.get(
        "latency_budget_ms", DEFAULT_EXPERIMENT["latency_budget_ms"]
    )
    latency_budget_us = int(latency_budget_ms * 1000) if latency_budget_ms else None

    warmup_runs = int(experiment_cfg.get("warmup_runs", DEFAULT_EXPERIMENT["warmup_runs"]))
    warmup_tensors = [
        _prepare_tensor(_download_image(bucket, images_prefix, img)) for img, _ in image_entries[:max_batch_size]
    ]
    warmup_time, warmup_download_time = _run_warmup(
        image_entries, warmup_runs, max_batch_size, bucket, images_prefix
    )

    report_samples = int(experiment_cfg.get("report_samples", DEFAULT_EXPERIMENT["report_samples"]))

    profiling_begin = datetime.datetime.now()
    (
        batch_profiles,
        stats,
        sample_predictions,
        batched_compute_time,
        batch_download_time,
    ) = _simulate_batches(
        image_entries, bucket, images_prefix, arrivals_us, max_batch_size, latency_budget_us, report_samples
    )
    profiling_end = datetime.datetime.now()

    first_arrival = arrivals_us[0]
    last_completion = batch_profiles[-1]["finish_time_us"] if batch_profiles else first_arrival
    simulated_span = max(last_completion - first_arrival, 1.0)
    stats["simulation_span_us"] = simulated_span
    throughput = stats["total_requests"] / (simulated_span / 1_000_000.0)

    download_time = batch_download_time + warmup_download_time
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
