import datetime
import json
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
    "batch_sizes": [1, 2, 4, 8],
    "image_limit": 128,
    "warmup_runs": 1,
    "repetitions": 2,
    "shuffle": False,
    "seed": 42,
    "report_predictions": 3,
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


def _load_images(
    bucket: str,
    image_prefix: str,
    entries: Sequence[Sequence[str]],
    limit: Optional[int],
    shuffle: bool,
    seed: int,
) -> List[str]:
    images = list(entries)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(images)
    if limit and limit > 0:
        images = images[:limit]
    paths = []
    for img, _label in images:
        download_path = os.path.join("/tmp", f"{uuid.uuid4()}-{os.path.basename(img)}")
        client.download(bucket, os.path.join(image_prefix, img), download_path)
        paths.append(download_path)
    if not paths:
        raise RuntimeError("No images available for profiling.")
    return paths


def _cleanup_images(paths: Sequence[str]):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _batch_arrays(arrays: Sequence[np.ndarray], batch_size: int) -> List[np.ndarray]:
    usable = (len(arrays) // batch_size) * batch_size
    if usable == 0:
        raise ValueError(f"Not enough images ({len(arrays)}) for batch size {batch_size}")
    batches = []
    for idx in range(0, usable, batch_size):
        chunk = arrays[idx : idx + batch_size]
        batches.append(np.stack(chunk, axis=0))
    return batches


def _timed_inference(batch: np.ndarray) -> Tuple[float, np.ndarray]:
    assert _session is not None and _session_input is not None and _session_output is not None
    begin = datetime.datetime.now()
    outputs = _session.run([_session_output], {_session_input: batch})
    end = datetime.datetime.now()
    logits = outputs[0]
    latency = (end - begin) / datetime.timedelta(microseconds=1)
    return latency, logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _format_predictions(batch: np.ndarray, logits: np.ndarray) -> List[Dict[str, Any]]:
    probabilities = _softmax(logits)
    formatted = []
    for probs in probabilities:
        top1_idx = int(np.argmax(probs))
        top5_idx = np.argsort(probs)[::-1][:5].tolist()
        formatted.append(
            {
                "label": idx2label[top1_idx],
                "confidence": float(probs[top1_idx]),
                "top5": [idx2label[idx] for idx in top5_idx],
            }
        )
    return formatted


def _profile_batch_size(
    batches: Sequence[np.ndarray],
    batch_size: int,
    warmup_runs: int,
    repetitions: int,
    report_predictions: int,
) -> Tuple[Dict[str, Any], float]:
    warmup = max(0, int(warmup_runs))
    for i in range(warmup):
        _timed_inference(batches[i % len(batches)])

    timings = []
    total_batches = 0
    sample_predictions: Optional[List[Dict[str, Any]]] = None
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

    profile: Dict[str, Any] = {
        "batch_size": batch_size,
        "batches": total_batches,
        "samples_processed": samples_processed,
        "total_latency_us": total_latency,
        "min_latency_us": float(min(timings)),
        "max_latency_us": float(max(timings)),
        "mean_latency_us": float(np.mean(timings)),
        "median_latency_us": float(np.median(timings)),
        "samples_per_second": throughput,
    }
    if sample_predictions:
        profile["sample_predictions"] = sample_predictions
    return profile, total_latency


def handler(event):
    bucket = event.get("bucket", {}).get("bucket")
    model_prefix = event.get("bucket", {}).get("model")
    images_prefix = event.get("bucket", {}).get("images")
    entries = event.get("object", {}).get("images", [])
    model_key = event.get("object", {}).get("model")

    if not bucket or not model_prefix or not images_prefix or not entries:
        raise ValueError("Missing storage configuration or image list in event payload.")

    model_download_time, model_process_time = _ensure_model(bucket, model_prefix, model_key)

    experiment_cfg = dict(DEFAULT_EXPERIMENT)
    experiment_cfg.update(event.get("experiment", {}))

    shuffle = bool(experiment_cfg.get("shuffle", False))
    seed = int(experiment_cfg.get("seed", DEFAULT_EXPERIMENT["seed"]))
    image_limit = experiment_cfg.get("image_limit", DEFAULT_EXPERIMENT["image_limit"])

    download_begin = datetime.datetime.now()
    image_paths = _load_images(bucket, images_prefix, entries, image_limit, shuffle, seed)
    download_end = datetime.datetime.now()

    tensors = [_prepare_tensor(path) for path in image_paths]
    _cleanup_images(image_paths)

    batch_sizes = experiment_cfg.get("batch_sizes", DEFAULT_EXPERIMENT["batch_sizes"])
    if not batch_sizes:
        raise ValueError("At least one batch size must be provided.")
    batch_sizes = [int(size) for size in batch_sizes]

    warmup_runs = int(experiment_cfg.get("warmup_runs", DEFAULT_EXPERIMENT["warmup_runs"]))
    repetitions = int(experiment_cfg.get("repetitions", DEFAULT_EXPERIMENT["repetitions"]))
    report_predictions = int(
        experiment_cfg.get("report_predictions", DEFAULT_EXPERIMENT["report_predictions"])
    )

    profiles = []
    total_latency = 0.0
    profiling_begin = datetime.datetime.now()
    for batch_size in batch_sizes:
        batches = _batch_arrays(tensors, batch_size)
        profile, latency = _profile_batch_size(
            batches, batch_size, warmup_runs, repetitions, report_predictions
        )
        profiles.append(profile)
        total_latency += latency
    profiling_end = datetime.datetime.now()

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    profiling_time = (profiling_end - profiling_begin) / datetime.timedelta(microseconds=1)

    return {
        "result": {
            "images_used": len(tensors),
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
