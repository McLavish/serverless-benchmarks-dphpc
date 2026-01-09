import datetime
import json
import os
import random
import statistics
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
    "batch_sizes": [1, 2, 4, 8],
    "num_sentences": 64,
    "warmup_runs": 1,
    "repetitions": 2,
    "shuffle": False,
    "seed": 42,
    "report_predictions": 3,
}

_session: Optional[ort.InferenceSession] = None
_tokenizer: Optional[Tokenizer] = None
_labels: Optional[Dict[int, str]] = None


def _ensure_model(bucket: str, model_prefix: str):
    """
    Lazily download and initialize the ONNX model and tokenizer.
    """
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _load_sentences(path: str, limit: Optional[int], shuffle: bool, seed: int) -> List[str]:
    with open(path, "r") as f:
        sentences = [json.loads(line)["text"] for line in f if line.strip()]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sentences)

    if limit is not None and limit > 0:
        sentences = sentences[:limit]
    return sentences


def _batch_sentences(sentences: Sequence[str], batch_size: int) -> List[List[str]]:
    usable = (len(sentences) // batch_size) * batch_size
    trimmed = sentences[:usable]
    return [trimmed[idx : idx + batch_size] for idx in range(0, usable, batch_size)]


def _timed_inference(batch_sentences: Sequence[str]) -> Tuple[float, np.ndarray]:
    inputs = _prepare_inputs(batch_sentences)
    begin = datetime.datetime.now()
    outputs = _session.run(None, inputs)
    end = datetime.datetime.now()
    latency = (end - begin) / datetime.timedelta(microseconds=1)
    logits = outputs[0]
    return latency, logits


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


def _profile_batch_size(
    sentences: Sequence[str],
    batch_size: int,
    warmup_runs: int,
    repetitions: int,
    report_predictions: int,
) -> Tuple[Dict[str, Any], float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batches = _batch_sentences(sentences, batch_size)
    if not batches:
        raise ValueError(
            f"Insufficient sentences ({len(sentences)}) for requested batch_size={batch_size}"
        )

    warmup = max(0, int(warmup_runs))
    for i in range(warmup):
        chunk = batches[i % len(batches)]
        _session.run(None, _prepare_inputs(chunk))

    timings = []
    sample_predictions: Optional[List[Dict[str, Any]]] = None
    total_batches = 0
    reps = max(1, int(repetitions))
    for _ in range(reps):
        for chunk in batches:
            latency, logits = _timed_inference(chunk)
            timings.append(latency)
            total_batches += 1
            if sample_predictions is None and report_predictions > 0:
                sample_predictions = _format_predictions(chunk, logits)[:report_predictions]

    total_latency = float(sum(timings))
    samples_processed = total_batches * batch_size
    throughput = samples_processed / total_latency * 1e6 if total_latency > 0 else 0.0

    summary: Dict[str, Any] = {
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
        summary["sample_predictions"] = sample_predictions
    return summary, total_latency


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

    num_sentences = experiment_cfg.get("num_sentences")
    num_sentences = int(num_sentences) if num_sentences is not None else None
    shuffle = bool(experiment_cfg.get("shuffle", False))
    seed = int(experiment_cfg.get("seed", DEFAULT_EXPERIMENT["seed"]))

    sentences = _load_sentences(text_download_path, num_sentences, shuffle, seed)
    os.remove(text_download_path)

    batch_sizes = experiment_cfg.get("batch_sizes", DEFAULT_EXPERIMENT["batch_sizes"])
    if not batch_sizes:
        raise ValueError("At least one batch size is required")
    batch_sizes = [int(size) for size in batch_sizes]

    warmup_runs = int(experiment_cfg.get("warmup_runs", DEFAULT_EXPERIMENT["warmup_runs"]))
    repetitions = int(experiment_cfg.get("repetitions", DEFAULT_EXPERIMENT["repetitions"]))
    report_predictions = int(
        experiment_cfg.get("report_predictions", DEFAULT_EXPERIMENT["report_predictions"])
    )

    profiling_begin = datetime.datetime.now()
    batch_profiles = []
    total_inference_time = 0.0
    for batch_size in batch_sizes:
        profile, latency = _profile_batch_size(
            sentences, batch_size, warmup_runs, repetitions, report_predictions
        )
        batch_profiles.append(profile)
        total_inference_time += latency
    profiling_end = datetime.datetime.now()

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    profiling_time = (profiling_end - profiling_begin) / datetime.timedelta(microseconds=1)

    return {
        "result": {
            "sentences_used": len(sentences),
            "batch_profiles": batch_profiles,
        },
        "measurement": {
            "download_time": download_time + model_download_time,
            "compute_time": profiling_time + model_process_time,
            "batched_compute_time": total_inference_time,
            "model_time": model_process_time,
            "model_download_time": model_download_time,
        },
    }
