import datetime
import json
import math
import os
import random
import tarfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import storage

client = storage.storage.get_instance()

MODEL_ARCHIVE = "bert-tiny-onnx.tar.gz"
MODEL_DIRECTORY = "/tmp/bert_language_model"
MODEL_SUBDIR = "bert-tiny-onnx"

DEFAULT_EXPERIMENT = {
    "max_batch_size": 16,
    "latency_budget_ms": 40,
    "arrival_rate_rps": 12,
    "simulation_duration_s": 5,
    "max_requests": 0,
    "warmup_runs": 1,
    "report_samples": 3,
    "shuffle": False,
    "seed": 42,
}

_session: Optional[ort.InferenceSession] = None
_tokenizer: Optional[Tokenizer] = None
_labels: Optional[Dict[int, str]] = None


def _ensure_model(bucket: str, model_prefix: str):
    global _session, _tokenizer, _labels

    model_path = os.path.join(MODEL_DIRECTORY, MODEL_SUBDIR)
    model_download_begin = datetime.datetime.now()
    model_download_end = model_download_begin

    if _session is None or _tokenizer is None or _labels is None:
        if not os.path.exists(model_path):
            os.makedirs(MODEL_DIRECTORY, exist_ok=True)
            archive_path = os.path.join("/tmp", f"{uuid.uuid4()}-{MODEL_ARCHIVE}")
            client.download(bucket, os.path.join(model_prefix, MODEL_ARCHIVE), archive_path)
            model_download_end = datetime.datetime.now()

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(MODEL_DIRECTORY)
            os.remove(archive_path)
        else:
            model_download_begin = datetime.datetime.now()
            model_download_end = model_download_begin

        model_process_begin = datetime.datetime.now()
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.enable_truncation(max_length=128)
        _tokenizer.enable_padding(length=128)

        label_map_path = os.path.join(model_path, "label_map.json")
        with open(label_map_path, "r") as f:
            raw_labels = json.load(f)
        _labels = {int(idx): label for idx, label in raw_labels.items()}

        onnx_path = os.path.join(model_path, "model.onnx")

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable (have: {available})")

        _session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
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


def _prepare_inputs(sentences: Sequence[str]):
    assert _tokenizer is not None

    encodings = _tokenizer.encode_batch(sentences)

    input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
    token_type_ids = np.array(
        [enc.type_ids if enc.type_ids else [0] * len(enc.ids) for enc in encodings],
        dtype=np.int64,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def _timed_inference(batch_sentences: Sequence[str]) -> Tuple[float, np.ndarray]:
    # calls _prepare_inputs and runs the model, measuring latency (ms)
    inputs = _prepare_inputs(batch_sentences)
    begin = datetime.datetime.now()
    outputs = _session.run(None, inputs)
    end = datetime.datetime.now()
    latency = (end - begin) / datetime.timedelta(microseconds=1)
    logits = outputs[0]
    return latency, logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _format_predictions(sentences: Sequence[str], logits: np.ndarray) -> List[Dict[str, Any]]:
    assert _labels is not None
    probabilities = _softmax(logits)
    formatted = []
    for sentence, probs in zip(sentences, probabilities):
        label_idx = int(np.argmax(probs))
        label = _labels.get(label_idx, str(label_idx))
        formatted.append(
            {
                "text": sentence,
                "label": label,
                "confidence": float(probs[label_idx]),
            }
        )
    return formatted


def _load_sentences(path: str, shuffle: bool, seed: int) -> List[str]:
    # read jsonl file sentences, optionally shuffle them.
    with open(path, "r") as f:
        sentences = [json.loads(line)["text"] for line in f if line.strip()]
    if not sentences:
        raise RuntimeError("Sentence corpus is empty")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sentences)
    return sentences


def _percentile(sorted_values: List[float], percentile: float) -> float:
    # helper math to compute percentiles and aggregates stats
    if not sorted_values:
        return 0.0
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 1:
        return float(sorted_values[-1])
    idx = (len(sorted_values) - 1) * percentile
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(sorted_values[int(idx)])
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    return float(lower_val + (upper_val - lower_val) * (idx - lower))


def _summary(values: List[float]) -> Dict[str, float]:
    # summaries used in the request-level metrics section
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
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


def _generate_arrivals(
    rate_rps: float, duration_s: float, limit: Optional[int], seed: int
) -> List[int]:
    """
    poisson-sampled arrival timestamps.

    : rate_rps: Poisson arrival rate in requests per second
    : duration_s: total simulation duration in seconds
    : limit: optional maximum number of arrivals to generate
    : seed: random seed for reproducibility
    : list of arrival timestamps in microseconds
    """
    if rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive")
    rng = random.Random(seed)
    timestamp = 0.0
    arrivals = []
    while timestamp < duration_s:
        # samples exponential inter-arrival times
        inter = rng.expovariate(rate_rps)
        timestamp += inter
        # accumulates until duration/limit
        if timestamp > duration_s:
            break
        arrivals.append(timestamp)
        if limit and len(arrivals) >= limit:
            break
    if not arrivals:
        arrivals.append(min(duration_s, 1.0))
    # converts timestamps into ms
    return [int(ts * 1_000_000) for ts in arrivals]


def _make_request(sentence: str, arrival_time: int, identifier: int) -> Dict[str, Any]:
    return {"sentence": sentence, "arrival_time": arrival_time, "id": identifier}


def _run_warmup(sentences: Sequence[str], runs: int, batch_cap: int) -> float:
    if runs <= 0:
        return 0.0
    warmup_sentences = list(sentences[: min(len(sentences), batch_cap)])
    if not warmup_sentences:
        warmup_sentences = [sentences[0]]
    total = 0.0
    for _ in range(runs):
        inputs = _prepare_inputs(warmup_sentences)
        begin = datetime.datetime.now()
        _session.run(None, inputs)
        end = datetime.datetime.now()
        total += (end - begin) / datetime.timedelta(microseconds=1)
    return total


def _simulate_batches(
    sentences: Sequence[str],
    arrivals_us: List[int],
    max_batch_size: int,
    latency_budget_us: Optional[int],
    report_samples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]], float]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    queue: List[Dict[str, Any]] = []
    current_time = 0.0
    arrival_idx = 0
    sentence_idx = 0
    total_requests = len(arrivals_us)
    if total_requests == 0:
        raise RuntimeError("No arrivals were generated for the simulation")

    batch_profiles: List[Dict[str, Any]] = []
    request_waits: List[float] = []
    response_times: List[float] = []
    # keep some sample predictions for reporting
    sample_predictions: List[Dict[str, Any]] = []
    batched_compute_time = 0.0
    request_id = 0
    batch_id = 0

    def next_sentence():
        nonlocal sentence_idx
        sentence = sentences[sentence_idx % len(sentences)]
        sentence_idx += 1
        return sentence

    # enqueues requests. flushes batches either when:
    # 1) the queue reaches max_batch_size, or 
    # 2) when the oldest arrival hits its latency budget
    while queue or arrival_idx < total_requests:
        if not queue:
            arrival_time = arrivals_us[arrival_idx]
            current_time = max(current_time, arrival_time)
            queue.append(_make_request(next_sentence(), arrival_time, request_id))
            request_id += 1
            arrival_idx += 1
            continue

        next_arrival_time = arrivals_us[arrival_idx] if arrival_idx < total_requests else None
        oldest_arrival = queue[0]["arrival_time"]
        deadline = (
            oldest_arrival + latency_budget_us
            if latency_budget_us is not None
            else float("inf")
        )

        should_flush = False
        trigger = None

        # tracks trigger reasons
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
            queue.append(_make_request(next_sentence(), next_arrival_time, request_id))
            request_id += 1
            arrival_idx += 1
            continue
        
        # runs timed inference for each batch, records queueing delays/response times
        batch_size = min(len(queue), max_batch_size)
        batch_requests = [queue.pop(0) for _ in range(batch_size)]
        start_time = max(current_time, batch_requests[-1]["arrival_time"])
        sentences_batch = [req["sentence"] for req in batch_requests]
        latency_us, logits = _timed_inference(sentences_batch)
        batched_compute_time += latency_us
        finish_time = start_time + latency_us

        waits = [start_time - req["arrival_time"] for req in batch_requests]
        responses = [finish_time - req["arrival_time"] for req in batch_requests]
        request_waits.extend(waits)
        response_times.extend(responses)

        # saves a few predictions for reporting
        formatted_preds = _format_predictions(sentences_batch, logits)
        if report_samples > 0 and len(sample_predictions) < report_samples:
            remaining = report_samples - len(sample_predictions)
            sample_predictions.extend(formatted_preds[:remaining])

        batch_profiles.append(
            {
                "batch_id": batch_id,
                "size": batch_size,
                "trigger": trigger,
                "start_time_us": float(start_time),
                "finish_time_us": float(finish_time),
                "inference_time_us": float(latency_us),
                "wait_min_us": float(min(waits)),
                "wait_max_us": float(max(waits)),
                "wait_avg_us": float(sum(waits) / len(waits)),
            }
        )
        batch_id += 1

        current_time = finish_time
        while arrival_idx < total_requests and arrivals_us[arrival_idx] <= current_time:
            queue.append(_make_request(next_sentence(), arrivals_us[arrival_idx], request_id))
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
    text_prefix = event.get("bucket", {}).get("text")
    text_key = event.get("object", {}).get("input")

    if not bucket or not model_prefix or not text_prefix or not text_key:
        raise ValueError("Missing storage configuration in event payload")

    download_begin = datetime.datetime.now()
    text_download_path = os.path.join("/tmp", f"{uuid.uuid4()}-{os.path.basename(text_key)}")
    client.download(bucket, os.path.join(text_prefix, text_key), text_download_path)
    download_end = datetime.datetime.now()

    model_download_time, model_process_time = _ensure_model(bucket, model_prefix)
    assert _session is not None and _labels is not None and _tokenizer is not None

    experiment_cfg = dict(DEFAULT_EXPERIMENT)
    experiment_cfg.update(event.get("experiment", {}))

    shuffle = bool(experiment_cfg.get("shuffle", False))
    seed = int(experiment_cfg.get("seed", DEFAULT_EXPERIMENT["seed"]))
    sentences = _load_sentences(text_download_path, shuffle, seed)
    os.remove(text_download_path)

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
    warmup_time = _run_warmup(sentences, warmup_runs, max_batch_size)

    report_samples = int(experiment_cfg.get("report_samples", DEFAULT_EXPERIMENT["report_samples"]))

    profiling_begin = datetime.datetime.now()
    batch_profiles, stats, sample_predictions, batched_compute_time = _simulate_batches(
        sentences, arrivals_us, max_batch_size, latency_budget_us, report_samples
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
